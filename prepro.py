from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import googleapiclient.discovery
import re
from datetime import datetime
import isodate  # For duration parsing
from abc import ABC, abstractmethod
import yt_dlp
import os
#for splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
#chroma for vector store and pincecone
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pinecone
from pinecone import Pinecone, ServerlessSpec
#embedding model and chatmodel
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") 
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',google_api_key=gemini_api_key)
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
parser=StrOutputParser()

api_key=os.getenv('YOUTUBE_API_KEY')






def load_transcript_file(filename):
    """Safely load transcript file with error handling"""
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found in directory: {os.getcwd()}")
            print(f"📁 Files in current directory: {os.listdir('.')}")
            return None
        
        # Read the file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Successfully loaded {filename} - {len(content)} characters")
        return content
        
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

# Fix your transcript loading
transcript_file = "transcript_liex_andrew.txt"
transcript = load_transcript_file(transcript_file)

if transcript is None:
    print("🚨 Cannot proceed - transcript file not found!")
    exit()

class YouTubeMetadataExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
        
        # YouTube category ID to name mapping
        self.category_map = {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles',
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '18': 'Short Movies',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '21': 'Videoblogging',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology',
            '29': 'Nonprofits & Activism',
            '30': 'Movies',
            '31': 'Anime/Animation',
            '32': 'Action/Adventure',
            '33': 'Classics',
            '34': 'Comedy',
            '35': 'Documentary',
            '36': 'Drama',
            '37': 'Family',
            '38': 'Foreign',
            '39': 'Horror',
            '40': 'Sci-Fi/Fantasy',
            '41': 'Thriller',
            '42': 'Shorts',
            '43': 'Shows',
            '44': 'Trailers'
        }
    
    def get_video_metadata(self, video_url):
        """Get useful metadata for RAG applications"""
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        
        try:
            # Request only necessary parts
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails,topicDetails",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return {"error": "Video not found or private"}
            
            return self._parse_rag_metadata(response['items'][0])
            
        except Exception as e:
            return {"error": f"API Error: {str(e)}"}
    
    def _parse_rag_metadata(self, item):
        """Parse only metadata useful for RAG applications"""
        snippet = item['snippet']
        statistics = item.get('statistics', {})
        content_details = item.get('contentDetails', {})
        topic_details = item.get('topicDetails', {})
        
        # Parse ISO 8601 duration to seconds
        duration_seconds = self._parse_duration(content_details.get('duration', 'PT0S'))
        
        # Convert category ID to category name
        category_id = snippet.get('categoryId', '')
        category_name = self.category_map.get(category_id, 'Unknown')
        
        # Extract useful topic information
        topic_categories = topic_details.get('topicCategories', [])
        simplified_topics = self._simplify_topics(topic_categories)
        
        metadata = {
            # === CORE IDENTIFICATION ===
            'video_id': item['id'],
            'url': f"https://www.youtube.com/watch?v={item['id']}",
            
            # === CONTENT INFORMATION ===
            'title': snippet['title'],
            
            'published_at': snippet['publishedAt'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'category': category_name,
            'tags': snippet.get('tags', []),
            
            # === ENGAGEMENT METRICS ===
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            
            # === CONTENT PROPERTIES ===
            'duration_seconds': duration_seconds,
            
            'caption_available': content_details.get('caption', 'false') == 'true',
            'licensed_content': content_details.get('licensedContent', False),
            
            # === TOPIC INFORMATION ===
            'topics': simplified_topics,
            
            
        }
        
        # Remove empty values to save space
        return self._clean_metadata(metadata)
    
    def _simplify_topics(self, topic_urls):
        """Extract simple topic names from Wikipedia URLs"""
        if not topic_urls:
            return []
        
        simplified = []
        for url in topic_urls:
            # Extract topic from Wikipedia URL format
            match = re.search(r'/([^/]+)$', url)
            if match:
                topic = match.group(1).replace('_', ' ').title()
                simplified.append(topic)
        return simplified
    
    def _clean_metadata(self, metadata):
        """Remove empty or useless values from metadata"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if value == 0 and key.endswith('_count'):
                # Keep zero counts as they might be meaningful
                cleaned[key] = value
            elif value != 0:
                cleaned[key] = value
        return cleaned
    
    def _extract_video_id(self, url):
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _parse_duration(self, duration):
        """Convert ISO 8601 duration to seconds"""
        try:
            return int(isodate.parse_duration(duration).total_seconds())
        except:
            # Fallback manual parsing
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return hours * 3600 + minutes * 60 + seconds
            return 0

    def get_category_name(self, category_id):
        """Convert category ID to human-readable name"""
        return self.category_map.get(str(category_id), 'Unknown')


# Get metadata
video_url='https://www.youtube.com/watch?v=0jspaMLxBig'
metadata_extractor = YouTubeMetadataExtractor(api_key)
metadata = metadata_extractor.get_video_metadata(video_url)

# makig chunks
def parse_timestamped_transcript(transcript_text):
    """Parse transcript with both MM:SS and HH:MM:SS timestamp formats"""
    lines = transcript_text.strip().split('\n')
    chunks = []
    current_timestamp = None
    
    print(f"🔍 Processing {len(lines)} total lines...")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check for BOTH timestamp formats
        timestamp_match = re.search(r'^(\d+:\d{2}:\d{2})$', line)  # HH:MM:SS first
        if not timestamp_match:
            timestamp_match = re.search(r'^(\d+:\d{2})$', line)    # Then MM:SS
        
        if timestamp_match:
            current_timestamp = timestamp_match.group(1)
            # Debug timestamp format changes
            if ":" in current_timestamp and current_timestamp.count(":") == 2:
                print(f"  🔄 Found HH:MM:SS format at line {i}: {current_timestamp}")
        elif current_timestamp:
            # This is a text line with the previous timestamp
            chunks.append({
                'timestamp': current_timestamp,
                'text': line,
                'timestamp_seconds': convert_to_seconds(current_timestamp)
            })
            current_timestamp = None
    
    print(f"✅ Parsed {len(chunks)} segments from transcript")
    return chunks
def convert_to_seconds(timestamp):
    """Convert MM:SS or HH:MM:SS to seconds"""
    parts = timestamp.split(':')
    
    if len(parts) == 3:  # HH:MM:SS format
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    else:
        print(f"⚠️  Unexpected timestamp format: {timestamp}")
        return 0
def create_smart_timestamp_chunks(timestamped_chunks, time_window):
    """Group chunks by time windows while preserving timestamps"""
    # If no segments, return empty list
    if not timestamped_chunks:
        return []
    
    chunks = []  # Final grouped chunks
    current_chunk = []  # Temporary storage for current group
    # Start with the first segment's timestamp
    current_start_time = timestamped_chunks[0]['timestamp_seconds']
    
    for segment in timestamped_chunks:
        # Check if this segment is within 60 seconds of our start time
        if segment['timestamp_seconds'] - current_start_time <= time_window:
            # If yes, add to current group
            current_chunk.append(segment)
        else:
            # If no, save the current group and start a new one
            if current_chunk:
                chunks.append(create_chunk_object(current_chunk))
            
            # Start new group with this segment
            current_chunk = [segment]
            current_start_time = segment['timestamp_seconds']
    
    # Don't forget the last group after loop ends
    if current_chunk:
        chunks.append(create_chunk_object(current_chunk))
    
    return chunks
def create_chunk_object(segments):
    """Create a chunk from multiple timestamped segments"""
    # Combine all segment texts into one continuous text
    chunk_text = ' '.join([seg['text'] for seg in segments])
    
    return {
        'text': chunk_text,                    # Combined text of all segments
        'start_time': segments[0]['timestamp'], # First segment's timestamp
        'start_seconds': segments[0]['timestamp_seconds'],
        'end_time': segments[-1]['timestamp'],  # Last segment's timestamp  
        'end_seconds': segments[-1]['timestamp_seconds'],
        'segment_count': len(segments),        # How many segments in this chunk
        'segments': segments  # Keep original segments for precise timestamp lookup
    }    
timestamped_transcript=parse_timestamped_transcript(transcript)
chunks=create_smart_timestamp_chunks(timestamped_transcript,120)
print(f"✅ Created {len(chunks)} chunks")

# FIX: Extract text from chunks before embedding
chunk_texts = [chunk['text'] for chunk in chunks]  # Extract just the text
print(f"📝 First chunk preview: {chunk_texts[0][:100]}...")
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)


# Convert the chunk_texts into sparse vectors

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunk_texts)
sparse_vectors = []
for i in range(len(chunk_texts)):
    row = tfidf_matrix[i]
    sparse_vectors.append({
        'indices': row.indices.tolist(),
        'values': row.data.tolist()
    })
# Extract the sparse embeddings data




# Now embed the TEXTS, not the dictionaries
print("🔧 Generating embeddings...")
splitted_dense_vector = dense_embeddings.embed_documents(chunk_texts)  # Use chunk_texts, not chunks
print(f"✅ Generated {len(splitted_dense_vector)} embeddings")
# Later, when storing in Pinecone:
#pinecone_metadata = metadata.copy()
#pinecone_metadata['text'] = transcript_text  # Add your transcript here

# Now pinecone_metadata is ready for upsert


vectors_to_upsert = []
        
    
    
pc = Pinecone(api_key=api_key)

index_name = "youtube-database"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
# Your existing setup

for i, (dense_vector, sparse_vector, chunk) in enumerate(zip(splitted_dense_vector, sparse_vectors, chunks)):
    chunk_metadata = metadata.copy()
    chunk_metadata.update({
        "text": chunk['text'],
        "chunk_index": i,
        "total_chunks": len(chunks),
        "start_time": chunk['start_time'],
        "end_time": chunk['end_time'],
        "start_seconds": chunk['start_seconds'],
        "end_seconds": chunk['end_seconds']
    })
    
    # ✅ FIX: Use zero-padding for proper numeric sorting
    chunk_id = f"{metadata['video_id']}_chunk_{i:04d}"  # This creates 0000, 0001, 0002, etc.
    

    vectors_to_upsert.append({
        "id": chunk_id,  # ✅ FIX: Use the generated chunk_id, not d[chunk_id]
        "values": dense_vector,  # Your dense embedding
        "sparse_values": {
            "indices": sparse_vector['indices'],
            "values": sparse_vector['values']
        },
        "metadata": chunk_metadata
    })
   
index = pc.Index(index_name)
namespace='TimestampedChunks'
# Upsert to Pinecone hybrid index
index.upsert(
    vectors=vectors_to_upsert,
    namespace=namespace  # Optional: add namespace if needed
)   

