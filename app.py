import streamlit as st
import numpy as np
import tensorflow as tf
from time import time
import os
import librosa
import requests
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import tempfile

try:
    from PIL import Image
except ImportError:
    import Image

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #1e3a8a 0%, #4c1d95 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #4c1d95 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #60a5fa !important;
        font-weight: bold;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* Card styling */
    .genre-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin: 15px 0;
    }
    
    .genre-card h3 {
        color: #60a5fa !important;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .genre-card p, .genre-card li {
        color: #e2e8f0 !important;
        line-height: 1.8;
    }
    
    /* Title container */
    .title-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 40px;
        border-radius: 25px;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        padding: 35px;
        border-radius: 25px;
        color: white;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        animation: pulse 2s infinite;
        box-shadow: 0 15px 50px rgba(139, 92, 246, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 15px 50px rgba(139, 92, 246, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 20px 60px rgba(139, 92, 246, 0.6); }
    }
    
    /* Info card for sidebar */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: white !important;
    }
    
    .info-card b {
        color: #60a5fa !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 18px 45px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
    }
    
    /* Upload section styling */
    .upload-section {
        background: rgba(255, 255, 255, 0.08);
        padding: 30px;
        border-radius: 20px;
        border: 2px dashed rgba(139, 92, 246, 0.5);
        backdrop-filter: blur(10px);
        text-align: center;
        margin: 20px 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        color: #60a5fa !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
        font-size: 18px;
        font-weight: 600;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(34, 197, 94, 0.2) !important;
        color: #86efac !important;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    /* Top 3 predictions styling */
    .prediction-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(236, 72, 153, 0.15));
        padding: 20px;
        border-radius: 15px;
        margin: 12px 0;
        border-left: 5px solid;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Horizontal rule */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent);
        margin: 30px 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Header Section
st.markdown("""
    <div class="title-container">
        <h1 style="color: white; font-size: 52px; margin: 0; text-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            🎵 Music Genre Classification
        </h1>
        <p style="color: rgba(255, 255, 255, 0.95); font-size: 20px; margin-top: 15px;">
            Powered by Deep Learning & Audio Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 style="color: #60a5fa; margin-top: 20px;">🎼 About the Model</h3>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
        <b>Dataset:</b> GTZAN Genre Collection<br>
        <b>Samples:</b> ~1000 audio files<br>
        <b>Format:</b> .wav & .mp3<br>
        <b>Feature:</b> MFCC (40 coefficients)
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #60a5fa; margin-top: 30px;">🎸 Supported Genres</h3>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    genres_list = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 
                   'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    for genre in genres_list:
        st.markdown(f'<p style="margin: 8px 0; color: #e2e8f0;">🎵 {genre}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 style="color: #60a5fa;">📊 Model Performance</h3>', unsafe_allow_html=True)
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric("Accuracy", "87.5%")
    with col_metric2:
        st.metric("Precision", "85.2%")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="genre-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #60a5fa;">🎼 How It Works</h3>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color: #e2e8f0; line-height: 2;">
        <b style="color: #a78bfa;">1.</b> <b>Upload</b> your .wav or .mp3 audio file<br>
        <b style="color: #a78bfa;">2.</b> <b>Extract</b> MFCC features automatically<br>
        <b style="color: #a78bfa;">3.</b> <b>Classify</b> using our trained CNN model<br>
        <b style="color: #a78bfa;">4.</b> <b>View</b> confidence scores and predictions
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lottie Animation
    anime1 = "https://assets10.lottiefiles.com/private_files/lf30_fjln45y5.json"
    anime1_json = load_lottieurl(anime1)
    if anime1_json:
        st_lottie(anime1_json, key='music', height=300)

with col2:
    st.markdown('<div class="genre-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #60a5fa;">🔬 Feature Extraction Process</h3>', unsafe_allow_html=True)
    
    with st.expander("📝 View MFCC Extraction Code"):
        code = '''
genres = {'blues': 0, 'classical': 1, 'country': 2, 
          'disco': 3, 'hiphop': 4, 'jazz': 5, 
          'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

for genre, genre_number in genres.items():
    for filename in os.listdir(f'path to {genre}'):
        songname = f'path to {genre}/{filename}'
        audio, sr = librosa.load(songname, res_type='kaiser_fast')
        mfcc_fea = np.mean(
            librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,
            axis=0
        )
        dataset.append(mfcc_fea)
        labels.append(genre_number)
        '''
        st.code(code, language='python')
    st.markdown('</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("---")
st.markdown("""
    <div class="upload-section">
        <h2 style="color: #a78bfa; margin-bottom: 20px; font-size: 32px;">
            🎧 Upload Your Audio File
        </h2>
    </div>
""", unsafe_allow_html=True)

col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])

with col_upload2:
    audio_file = st.file_uploader(
        "Choose a .wav or .mp3 file",
        type=['wav', 'mp3'],
        help="Upload a .wav or .mp3 audio file for genre classification"
    )
    
    if audio_file is not None:
        # Display audio player with correct format
        file_extension = audio_file.name.split('.')[-1].lower()
        audio_format = f'audio/{file_extension}'
        st.audio(audio_file, format=audio_format)
        st.success(f"✅ {file_extension.upper()} file uploaded successfully!")

# Prediction Section
if audio_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button('🎯 Classify Genre', use_container_width=True)
    
    if predict_button:
        try:
            with st.spinner('🎵 Analyzing audio features...'):
                mydict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 
                         'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 
                         'reggae': 8, 'rock': 9}
                
                # Create a temporary file to save the uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Feature extraction - librosa handles both WAV and MP3
                librosa_audio_data, librosa_sample_rate = librosa.load(tmp_file_path, res_type='kaiser_fast')
                mfccs = np.mean(
                    librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40).T,
                    axis=0
                )
                
                x = []
                x.append(mfccs)
                x = np.array(x)
                x = np.reshape(x, (x.shape[0], 10, 4, 1))
                
                # Model prediction
                model = tf.keras.models.load_model("gerne_model.h5")
                y_pre = model.predict(x)
                
                # Get prediction probabilities
                predicted_genre_idx = np.argmax(y_pre[0])
                confidence_scores = y_pre[0]
                
                # Find genre name
                predicted_genre = [genre for genre, idx in mydict.items() if idx == predicted_genre_idx][0]
                confidence = confidence_scores[predicted_genre_idx] * 100
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            st.markdown("---")
            
            # Prediction Result
            st.markdown(f"""
                <div class="prediction-box">
                    🎵 Predicted Genre: {predicted_genre.upper()} 🎵<br>
                    <span style="font-size: 24px; opacity: 0.95;">Confidence: {confidence:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed Results
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown('<div class="genre-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #60a5fa;">📊 Confidence Scores</h3>', unsafe_allow_html=True)
                
                # Create bar chart
                genre_names = [genre for genre, _ in sorted(mydict.items(), key=lambda x: x[1])]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=genre_names,
                        y=confidence_scores * 100,
                        marker=dict(
                            color=confidence_scores * 100,
                            colorscale='Plasma',
                            showscale=True,
                            line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
                        ),
                        text=[f'{score*100:.1f}%' for score in confidence_scores],
                        textposition='outside',
                        textfont=dict(size=12, color='white')
                    )
                ])
                
                fig.update_layout(
                    title={
                        'text': "Genre Probability Distribution",
                        'font': {'size': 18, 'color': '#e2e8f0'}
                    },
                    xaxis_title="Genres",
                    yaxis_title="Confidence (%)",
                    height=400,
                    template="plotly_dark",
                    showlegend=False,
                    plot_bgcolor='rgba(0, 0, 0, 0.2)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_res2:
                st.markdown('<div class="genre-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #60a5fa;">🎯 Top 3 Predictions</h3>', unsafe_allow_html=True)
                
                # Get top 3 predictions
                top_3_idx = np.argsort(confidence_scores)[::-1][:3]
                
                colors = ['#8b5cf6', '#ec4899', '#f59e0b']
                
                for rank, idx in enumerate(top_3_idx, 1):
                    genre_name = [genre for genre, g_idx in mydict.items() if g_idx == idx][0]
                    score = confidence_scores[idx] * 100
                    
                    st.markdown(f"""
                        <div class="prediction-card" style="border-left-color: {colors[rank-1]};">
                            <span style="font-size: 28px; font-weight: bold; color: {colors[rank-1]};">#{rank}</span>
                            <span style="font-size: 22px; margin-left: 20px; color: #e2e8f0; font-weight: 600;">{genre_name.upper()}</span>
                            <span style="float: right; font-size: 20px; color: {colors[rank-1]}; font-weight: bold;">{score:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown('<div class="genre-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #60a5fa;">📈 Analysis Metrics</h3>', unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Sample Rate", f"{librosa_sample_rate} Hz")
                    st.metric("Duration", f"{len(librosa_audio_data)/librosa_sample_rate:.1f}s")
                with col_m2:
                    st.metric("MFCC Features", "40")
                    st.metric("File Format", audio_file.name.split('.')[-1].upper())
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 **Tip for MP3 files:** Make sure you have ffmpeg installed on your system for MP3 support.\n\n"
                   "**Installation:**\n"
                   "- Linux: `sudo apt-get install ffmpeg`\n"
                   "- Mac: `brew install ffmpeg`\n"
                   "- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: rgba(255, 255, 255, 0.8); padding: 25px;">
        <p style="font-size: 16px; line-height: 1.8;">
            Built with ❤️ using <b style="color: #60a5fa;">Streamlit</b>, 
            <b style="color: #ec4899;">TensorFlow</b> & <b style="color: #8b5cf6;">Librosa</b><br>
            <span style="font-size: 14px; opacity: 0.8;">GTZAN Dataset | Deep Learning Classification Model | Supports WAV & MP3</span>
        </p>
    </div>
""", unsafe_allow_html=True)