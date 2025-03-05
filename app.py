import streamlit as st
import torch
import torch.nn as nn
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Nazm Generator",
    page_icon="📝",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .generated-nazm {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
        color: #111827;  /* Darker text color */
        font-size: 1.2rem;
        line-height: 1.6;
        white-space: pre-line;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ScaleLayer from the original model
class ScaleLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
    def forward(self, x):
        return x * self.scale

# RMSNorm from the original model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)

# ROPE Functions
def get_rope_embeddings(seq_len, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()  
    sinusoid_inp = torch.outer(positions, inv_freq)           
    sin = torch.sin(sinusoid_inp)  
    cos = torch.cos(sinusoid_inp)  
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
    return sin, cos

def apply_rope(x):
    batch, seq_len, dim = x.shape
    sin, cos = get_rope_embeddings(seq_len, dim, x.device)
    sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(0)
    
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    sin_half = sin[..., ::2]
    cos_half = cos[..., ::2]
    
    x1_rot = x1 * cos_half - x2 * sin_half
    x2_rot = x1 * sin_half + x2 * cos_half
    x_rot = torch.stack((x1_rot, x2_rot), dim=-1).reshape(batch, seq_len, dim)
    return x_rot

# Nazm Generator Model
class NazmGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(NazmGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Use <PAD> index
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.lms_norm = RMSNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            RMSNorm(512),
            nn.Linear(512, vocab_size)
        )

    def forward(self, x):
        emb = self.embedding(x)
        emb = apply_rope(emb)
        lstm_out, _ = self.lstm(emb)
        last_out = lstm_out[:, -1, :]
        normed_out = self.lms_norm(last_out)
        logits = self.fc(normed_out)
        return logits

# Utility functions for preprocessing and generation
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' [NEWLINE] ')
    text = ''.join([c for c in text if c.isalpha() or c in [" ", "'", ".", "[", "]"]])
    return text

# Load resources
@st.cache_resource
def load_model_and_dicts():
    try:
        # Load dictionaries
        with open('word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
        
        with open('idx2word.pkl', 'rb') as f:
            idx2word = pickle.load(f)
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = NazmGeneratorModel(
            vocab_size=len(word2idx),
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2
        )
        
        # Load model weights
        model.load_state_dict(torch.load('urdu_poetry_gru.pth', map_location=device))
        model.eval()
        
        return model, word2idx, idx2word, device
    
    except Exception as e:
        st.error(f"Error loading model or dictionaries: {e}")
        return None, None, None, None

def generate_nazm(model, prompt, word2idx, idx2word, device, sequence_length=50, gen_length=100, temperature=1.0):
    model.eval()
    
    # Preprocess prompt
    tokens = preprocess_text(prompt).split()
    
    # Convert to indices
    seq_indices = [word2idx.get(token, word2idx.get('<UNK>', 0)) for token in tokens]
    
    # Pad or trim sequence
    if len(seq_indices) < sequence_length:
        seq_indices = [word2idx['<PAD>']] * (sequence_length - len(seq_indices)) + seq_indices
    else:
        seq_indices = seq_indices[-sequence_length:]
    
    # Convert to tensor
    input_tensor = torch.LongTensor(seq_indices).unsqueeze(0).to(device)
    
    # Generate text
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(gen_length):
            # Ensure current sequence has required length
            if len(seq_indices) < sequence_length:
                current_tokens = [word2idx['<PAD>']] * (sequence_length - len(seq_indices)) + seq_indices
            else:
                current_tokens = seq_indices[-sequence_length:]
            
            # Convert to tensor
            input_tensor = torch.LongTensor(current_tokens).unsqueeze(0).to(device)
            
            # Get prediction with temperature
            logits = model(input_tensor)
            # Apply temperature
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.multinomial(probabilities, 1).item()
            
            # Get predicted word
            predicted_word = idx2word.get(predicted_idx, '<UNK>')
            
            # Update sequences
            generated.append(predicted_word)
            seq_indices.append(predicted_idx)
            
            # Stop if end token or max length reached
            if predicted_word == '<END>' or len(generated) > gen_length:
                break
    
    # Format generated text
    nazm = ' '.join(generated)
    
    # Add line breaks for readability
    formatted_nazm = ''
    line_length = 0
    for word in nazm.split():
        if line_length > 40:
            formatted_nazm += '\n'
            line_length = 0
        formatted_nazm += word + ' '
        line_length += len(word) + 1
    
    return formatted_nazm.replace('[NEWLINE]', '\n')

# Main Streamlit App
def main():
    # App title and description
    st.markdown("<h1 class='main-header'>Nazm Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter a sentence and generate a beautiful nazm</p>", unsafe_allow_html=True)

    # Load model and dictionaries
    model, word2idx, idx2word, device = load_model_and_dicts()

    # Input text box for the sentence
    input_sentence = st.text_area("Enter a sentence to generate a nazm:", height=100)
    
    # Add controls for generation parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, 
                             help="Higher values produce more diverse but potentially less coherent text")
    with col2:
        gen_length = st.slider("Generation Length", min_value=20, max_value=200, value=100, step=10,
                            help="Number of words to generate")

    # Generate button
    if st.button("Generate Nazm"):
        if not input_sentence:
            st.warning("Please enter a sentence first.")
        elif model is None or word2idx is None or idx2word is None:
            st.error("Model or dictionaries failed to load. Please check your files.")
        else:
            with st.spinner("Generating nazm..."):
                # Generate the nazm with user-selected parameters
                nazm = generate_nazm(model, input_sentence, word2idx, idx2word, device, 
                                     gen_length=gen_length, temperature=temperature)
                
                # Display the generated nazm
                st.markdown("<h2 class='sub-header'>Generated Nazm:</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='generated-nazm'>{nazm}</div>", unsafe_allow_html=True)

    # Information about the model
    with st.expander("About this Model"):
        st.write("""
        This application uses a pre-trained LSTM language model to generate Urdu poetry (nazm) 
        based on your input sentence. The model is enhanced with advanced techniques like 
        Rotary Positional Embeddings (ROPE) and Root Mean Square Layer Normalization (RMSNorm).
        
        Note: The quality of the generated nazm depends on the training data and model architecture.
        """)

    # Footer
    st.markdown("<div class='footer'>Created with ❤️ using Streamlit and PyTorch</div>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()