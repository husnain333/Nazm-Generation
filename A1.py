import streamlit as st
import torch
import torch.nn as nn
import pickle

########################################
# 1. Load Vocabulary and Model
########################################
@st.cache(allow_output_mutation=True)
def load_model():
    # Load vocabulary
    with open("vocab.pkl", "rb") as f:
        word2idx = pickle.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)
    
    ########################################
    # 2. Define the Model Architecture
    ########################################
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-8):
            """
            RMSNorm normalizes inputs by their root-mean-square.
            """
            super(RMSNorm, self).__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            # x shape: (..., dim)
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.scale * (x / rms)
    # ROPE (Rotary Positional Embedding) Functions
    ########################################
    def get_rope_embeddings(seq_len, dim, device):
        """
        Compute sin and cos embeddings for ROPE.
        Assumes dim is even.
        """
        # Compute inverse frequency for each even dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).float()  
        sinusoid_inp = torch.outer(positions, inv_freq)           
        sin = torch.sin(sinusoid_inp)  
        cos = torch.cos(sinusoid_inp)  
        # Expand to match original dim by interleaving sin and cos along last dimension.
        # One simple approach is to repeat each column twice.
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
        return sin, cos
    def apply_rope(x):
        """
        Applies Rotary Positional Embedding (ROPE) to input x.
        x: Tensor of shape (batch, seq_len, dim) where dim is even.
        """
        batch, seq_len, dim = x.shape
        # Get sin and cos for the current sequence length
        sin, cos = get_rope_embeddings(seq_len, dim, x.device)  # each: (seq_len, dim)
        # Expand to match batch size
        sin = sin.unsqueeze(0)  # (1, seq_len, dim)
        cos = cos.unsqueeze(0)  # (1, seq_len, dim)
        
        # ROPE is typically applied per pair of dimensions.
        # Here we split the last dim into even and odd indices:
        x1 = x[..., ::2]  # (batch, seq_len, dim/2)
        x2 = x[..., 1::2] # (batch, seq_len, dim/2)
        
        # Similarly for sin and cos (also split in half)
        sin_half = sin[..., ::2]
        cos_half = cos[..., ::2]
        
        # Apply the rotation
        x1_rot = x1 * cos_half - x2 * sin_half
        x2_rot = x1 * sin_half + x2 * cos_half
        # Reconstruct interleaved tensor
        x_rot = torch.stack((x1_rot, x2_rot), dim=-1).reshape(batch, seq_len, dim)
        return x_rot
    # Pretrained LSTM Poetry Model with ROPE and RMSNorm
    ########################################
    class PretrainedPoetryLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
            """
            A poetry language model using a pretrained (unsupervised) LSTM,
            enhanced with ROPE and RMSNorm.
            """
            super(PretrainedPoetryLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
            # LSTM is bidirectional. (You can change to uni-directional if desired.)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                                batch_first=True, dropout=0.1, bidirectional=True)
            # RMSNorm applied on the output (hidden dim doubled due to bidirectionality)
            self.lms_norm = RMSNorm(hidden_dim * 2)
            # Fully connected layers for prediction
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                RMSNorm(512),
                nn.Linear(512, vocab_size)
            )
        def forward(self, x):
            # x: (batch, seq_len) integer token indices
            emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
            # Apply ROPE to the embeddings
            emb = apply_rope(emb)
            # Pass embeddings through LSTM
            lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim*2)
            # Here, we take the last time-step’s output.
            last_out = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
            # Normalize using RMSNorm
            normed_out = self.lms_norm(last_out)
            # Fully connected layers produce logits for next-token prediction.
            logits = self.fc(normed_out)
            return logits

    # Initialize the model and load the saved weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedPoetryLSTM(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2)
    model.load_state_dict(torch.load("urdu_poetry.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, word2idx, idx2word, device

model, word2idx, idx2word, device = load_model()

########################################
# 3. Preprocessing & Generation Functions
########################################
def preprocess_prompt(prompt):
    """Preprocess the prompt similar to training."""
    prompt = prompt.lower()
    prompt = prompt.replace('\n', ' [NEWLINE] ')
    # Retain only alphabets and allowed symbols.
    prompt = ''.join([c for c in prompt if c.isalpha() or c in [" ", "'", ".", "[", "]"]])
    return prompt.split()

def generate_text(prompt, gen_length=20, sequence_length=20):
    """
    Generate a nazm (poetry) by iteratively predicting the next word.
    
    Args:
        prompt (str): Starting text.
        gen_length (int): Number of words to generate.
        sequence_length (int): Fixed input length for the model.
    
    Returns:
        str: The full generated nazm.
    """
    tokens = preprocess_prompt(prompt)
    generated = tokens.copy()  # Start with the prompt tokens
    
    model.eval()
    with torch.no_grad():
        for _ in range(gen_length):
            # Prepare current sequence: pad if needed or use the last sequence_length tokens.
            if len(generated) < sequence_length:
                current_tokens = ['<PAD>'] * (sequence_length - len(generated)) + generated
            else:
                current_tokens = generated[-sequence_length:]
            
            # Convert tokens to indices.
            seq_indices = [word2idx.get(token, word2idx['<UNK>']) for token in current_tokens]
            input_tensor = torch.LongTensor(seq_indices).unsqueeze(0).to(device)  # Shape: (1, sequence_length)
            
            # Predict next word.
            logits = model(input_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_word = idx2word.get(predicted_idx, '<UNK>')
            
            # Append the predicted word.
            generated.append(predicted_word)
    
    return " ".join(generated)

########################################
# 4. Streamlit UI
########################################
st.title("Urdu Poetry (Nazm) Generator")
st.write("Enter a line of poetry and let the model generate a nazm.")

input_line = st.text_input("Enter your line here:")

if st.button("Generate Nazm"):
    if input_line.strip():
        # You can adjust gen_length to control how many words to generate.
        output = generate_text(input_line, gen_length=20, sequence_length=20)
        st.subheader("Generated Nazm:")
        st.write(output)
    else:
        st.error("Please enter a valid line!")
