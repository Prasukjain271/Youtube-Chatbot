from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import googleapiclient.discovery
import re
from datetime import datetime
import isodate  # For duration parsing
from abc import ABC, abstractmethod
import yt_dlp
import os
from typing import List, Dict  
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

class TranscriptLoader:
    """
    The 'Librarian' Class.
    Responsible solely for finding, validating, and reading text files.
    """
    
    def __init__(self, base_directory="."):
        """
        Initialize the loader. 
        Args:
            base_directory (str): The folder where transcripts are usually kept. 
                                  Defaults to current directory (".").
        """
        self.base_directory = base_directory

    def load(self, filename):
        """
        Safely loads a transcript file.
        Returns the text content or None if it fails.
        """
        # Combine base dir and filename to get full path
        full_path = os.path.join(self.base_directory, filename)

        try:
            # 1. Validation Check
            if not os.path.exists(full_path):
                print(f"❌ File not found: {full_path}")
                print(f"📁 Files in {self.base_directory}: {os.listdir(self.base_directory)}")
                return None
            
            # 2. Reading
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 3. Success
            print(f"✅ Successfully loaded {filename} - {len(content)} characters")
            return content

        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None

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
        
        raw_tags = snippet.get('tags', [])
        normalized_tags = list(set([t.lower().strip() for t in raw_tags])) if raw_tags else []
        
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
            'tags': normalized_tags,
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




from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def parse_mixed_timestamps(raw_text: str) -> List[Dict]:
    """
    Parse transcripts with both MM:SS and HH:MM:SS timestamp formats
    """
    # Regex patterns for both timestamp formats
    mm_ss_pattern = r'^(\d{1,2}):(\d{2})'  # MM:SS
    hh_mm_ss_pattern = r'^(\d{1,2}):(\d{2}):(\d{2})'  # HH:MM:SS
    
    entries = []
    lines = raw_text.strip().split('\n')
    
    current_time = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for HH:MM:SS format first (more specific)
        hh_match = re.match(hh_mm_ss_pattern, line)
        if hh_match:
            # Save previous entry if exists
            if current_time is not None and current_text:
                text = ' '.join(current_text).strip()
                if text:
                    entries.append({
                        'text': text,
                        'start': current_time,
                        'duration': 0
                    })
            
            # Parse HH:MM:SS
            hours, minutes, seconds = map(int, hh_match.groups())
            current_time = hours * 3600 + minutes * 60 + seconds
            current_text = [line[len(hh_match.group(0)):].strip()]
            continue
            
        # Check for MM:SS format
        mm_match = re.match(mm_ss_pattern, line)
        if mm_match:
            # Save previous entry if exists
            if current_time is not None and current_text:
                text = ' '.join(current_text).strip()
                if text:
                    entries.append({
                        'text': text,
                        'start': current_time,
                        'duration': 0
                    })
            
            # Parse MM:SS
            minutes, seconds = map(int, mm_match.groups())
            current_time = minutes * 60 + seconds
            current_text = [line[len(mm_match.group(0)):].strip()]
            continue
            
        # If no timestamp, it's continuation text
        if current_text is not None:
            current_text.append(line)
    
    # Don't forget the last entry
    if current_time is not None and current_text:
        text = ' '.join(current_text).strip()
        if text:
            entries.append({
                'text': text,
                'start': current_time,
                'duration': 0
            })
    
    # Calculate durations
    return calculate_durations(entries)

def calculate_durations(entries: List[Dict]) -> List[Dict]:
    """Calculate durations between entries"""
    if not entries:
        return entries
    
    # Calculate durations for all but last entry
    for i in range(len(entries) - 1):
        entries[i]['duration'] = entries[i + 1]['start'] - entries[i]['start']
    
    # Estimate last entry duration (usually similar to previous ones)
    if len(entries) > 1:
        avg_duration = sum(entry['duration'] for entry in entries[:-1]) / (len(entries) - 1)
        entries[-1]['duration'] = max(3, min(avg_duration, 10))  # Reasonable range
    else:
        entries[-1]['duration'] = 5  # Default for single entry
    
    return entries

# def get_optimal_threshold(video_duration_minutes, content_type="conversation"):
#     """
#     Calculate optimal similarity threshold based on video length
#     """
#     base_thresholds = {
#         "conversation": {
#             "short": (0, 30, 0.4),    # <30min: 0.4
#             "medium": (30, 90, 0.25),  # 30-90min: 0.25  
#             "long": (90, 180, 0.12),   # 90-180min: 0.15***
#              
#         },
#         "lecture": {
#             "short": (0, 30, 0.3),
#             "medium": (30, 90, 0.2),
#             "long": (90, 180, 0.12),
#             "very_long": (180, 9999, 0.08)
#         },
#         "documentary": {
#             "short": (0, 30, 0.35),
#             "medium": (30, 90, 0.22),
#             "long": (90, 180, 0.14),
#             "very_long": (180, 9999, 0.1)
#         }
#     }
    
#     config = base_thresholds[content_type]
    
#     for category, (min_dur, max_dur, threshold) in config.items():
#         if min_dur <= video_duration_minutes < max_dur:
#             return threshold
    
#     return 0.15  # Default fallback
def manual_semantic_chunking(transcript_entries, similarity_threshold=0.12):
    """
    Manual implementation of semantic chunking
    """
    # Use your GPU
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        # Get embeddings for each transcript entry
        texts = [entry['text'] for entry in transcript_entries]
        text_embeddings = embeddings_model.embed_documents(texts)
        
        chunks = []
        current_chunk = [transcript_entries[0]]
        
        for i in range(1, len(transcript_entries)):
            # Calculate similarity between current and previous
            sim = cosine_similarity(
                [text_embeddings[i-1]], 
                [text_embeddings[i]]
            )[0][0]
            
            if sim < similarity_threshold:
                # Low similarity = topic shift, create new chunk
                if current_chunk:
                    chunk_text = " ".join([entry['text'] for entry in current_chunk])
                    chunks.append({
                        'text': chunk_text,
                        'start_time': current_chunk[0]['start'],
                        'end_time': current_chunk[-1]['start'] + current_chunk[-1]['duration']
                    })
                current_chunk = [transcript_entries[i]]
            else:
                # High similarity = same topic, continue chunk
                current_chunk.append(transcript_entries[i])
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join([entry['text'] for entry in current_chunk])
            chunks.append({
                'text': chunk_text,
                'start_time': current_chunk[0]['start'],
                'end_time': current_chunk[-1]['start'] + current_chunk[-1]['duration']
            })
        
        return chunks
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def two_level_summarization(chunks, chunks_per_section=3):
    """
    Efficient approach: Group chunks → Section summaries → Final summary
    """
    # Step 1: Group chunks into logical sections
    sections = group_chunks_into_sections(chunks, chunks_per_section)
    
    # Step 2: Generate section summaries
    section_summaries = []
    for i, section_chunks in enumerate(sections):
        print(f"Summarizing section {i+1}/{len(sections)}...")
        section_summary = summarize_section(section_chunks)
        section_summaries.append(section_summary)
   
    # Step 3: Generate final comprehensive summary
    final_summary = create_final_summary(section_summaries)
    
    return final_summary, section_summaries

def group_chunks_into_sections(chunks, chunks_per_section=3):
    """
    Group 3 chunks together to form logical sections
    """
    sections = []
    for i in range(0, len(chunks), chunks_per_section):
        section_chunks = chunks[i:i + chunks_per_section]
        sections.append(section_chunks)
    return sections

def summarize_section(section_chunks):
    """
    Summarize a group of 3 related chunks
    """
    # Combine all chunk texts in the section
    section_text = "\n\n".join([
        f"[Part {j+1}, {chunk['start_time']}-{chunk['end_time']}s]: {chunk['text']}"
        for j, chunk in enumerate(section_chunks)
    ])
    
  
    prompt=PromptTemplate(template=f"""
    Summarize this section from a podcast conversation. This section contains {len(section_chunks)} sequential parts.

    SECTION CONTENT:
    {section_text}

    Provide a comprehensive 5-15 sentence summary that captures:
    - The main topics discussed across these parts
    - Key stories or examples shared
    - Important insights or arguments made
    - How the conversation evolves through this section

    Focus on the overall narrative flow rather than individual parts.
    """,
    input_variables=["section_text","section_chunks"])
    chain=prompt|llm|parser
    result=chain.invoke({"section_text":section_text,"section_chunks":section_chunks})
    
    return result

def create_final_summary(section_summaries):
    """section_text
    Create final video summary from section summaries
    """
    all_sections = "\n\n".join([
        f"=== Section {i+1} ===\n{summary}"
        for i, summary in enumerate(section_summaries)
    ])
    
   
    
    prompt=PromptTemplate(template= f"""
    Create a comprehensive summary of this entire podcast based on the section summaries below.

    SECTION SUMMARIES:
    {all_sections}

    Structure your final summary as:

    ## Podcast Overview
    - Main themes and topics covered
    - Overall conversation flow and structure

    ## Key Discussion Areas
    - Major subjects discussed (list 4-5 main areas)
    - Important insights and takeaways for each area

    ## Notable Stories & Examples
    - Memorable anecdotes shared by the guest
    - Practical examples and case studies mentioned

    ## Conclusion & Main Takeaways
    - Overall conclusions from the conversation
    - Key lessons for the audience

    Make it engaging and informative, capturing the essence of the entire conversation.
    """,
    input_variables=["all_sections"])
    chain=prompt|llm|parser
    result=chain.invoke({"all_sections":all_sections})
    
    return result

#get trnascript
loader = TranscriptLoader()         

transcript_file = "oops_project/joe_1.txt"
transcript = loader.load(transcript_file)
# get metadata
video_url='https://www.youtube.com/watch?v=BEWz4SXfyCQ'

metadata_extractor = YouTubeMetadataExtractor(api_key)
metadata = metadata_extractor.get_video_metadata(video_url)
#get summaries
transcript_with_structureddic=parse_mixed_timestamps(transcript)
chunks=manual_semantic_chunking(transcript_with_structureddic)   
final_summary, section_summaries=two_level_summarization(chunks)

links_from_andrew=["https://www.youtube.com/watch?v=iT8W6kaD-RA","https://www.youtube.com/watch?v=SOo4yNoaAoc&t=1s","https://www.youtube.com/watch?v=hMzfGZnaPN8","https://www.youtube.com/watch?v=25PtptE7mWk","https://www.youtube.com/watch?v=ZtTUfMHuioA",""]
links_from_liex=["https://www.youtube.com/watch?v=Qp0rCU49lMs","https://www.youtube.com/watch?v=m_CFCyc2Shs","https://www.youtube.com/watch?v=o3gbXDjNWyI","https://www.youtube.com/watch?v=7OLVwZeMCfY","https://www.youtube.com/watch?v=JN3KPFbWCy8"]
links_from_joerogan=["https://www.youtube.com/watch?v=BEWz4SXfyCQ","https://www.youtube.com/watch?v=hBMoPUAeLnY","https://www.youtube.com/watch?v=efs3QRr8LWw","https://www.youtube.com/watch?v=RcYjXbSJBN8","https://www.youtube.com/watch?v=jdVso9FSkmE"]
links_from_xxx=["https://www.youtube.com/watch?v=mPZkdNFkNps"]




api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)


# Convert the chunk_texts into sparse vectors

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(section_summaries)
sparse_vectors = []
for i in range(len(section_summaries)):
    row = tfidf_matrix[i]
    sparse_vectors.append({
        'indices': row.indices.tolist(),
        'values': row.data.tolist()
    })
# Extract the sparse embeddings data




# Now embed the TEXTS, not the dictionaries
print("🔧 Generating embeddings...")
splitted_dense_vector = dense_embeddings.embed_documents(section_summaries)  # Use chunk_texts, not chunks
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

for i, (dense_vector, sparse_vector, section) in enumerate(zip(splitted_dense_vector, sparse_vectors, section_summaries)):
    section_metadata = metadata.copy()
    section_metadata.update({
        "text": section,
        "chunk_index": i,
        "total_chunks": len(chunks),
       
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
        "metadata": section_metadata
    })
   
index = pc.Index(index_name)
namespace='sectionsummary' 
# Upsert to Pinecone hybrid index
index.upsert(
    vectors=vectors_to_upsert,
    namespace=namespace  # Optional: add namespace if needed
)   






from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform([final_summary])
sparse_vectors = []
for i in range(tfidf_matrix.shape[0]):
    row = tfidf_matrix[i]
    sparse_vectors.append({
        'indices': row.indices.tolist(),
        'values': row.data.tolist()
    })

splitted_dense_vector = dense_embeddings.embed_documents(final_summary) # Use chunk_texts, not chunks

vectors_to_upsert = []

for i, (dense_vector, sparse_vector, summary) in enumerate(zip(splitted_dense_vector, sparse_vectors, final_summary)):
    summary_metadata = metadata.copy()
    summary_metadata.update({
        "text": final_summary,
        "chunk_index": i,
        
       
    })
    
    # ✅ FIX: Use zero-padding for proper numeric sorting
     # This creates 0000, 0001, 0002, etc.
    chunk_id = f"{metadata['video_id']}_chunk_{i:04d}"  # This creates 0000, 0001, 0002, etc.


    vectors_to_upsert.append({
        "id": chunk_id,  # ✅ FIX: Use the generated chunk_id, not d[chunk_id]
        "values": dense_vector,  # Your dense embedding
        "sparse_values": {
            "indices": sparse_vector['indices'],
            "values": sparse_vector['values']
        },
        "metadata": summary_metadata
    })
   
index = pc.Index(index_name)
namespace='finalsummary'
# Upsert to Pinecone hybrid index
index.upsert(
    vectors=vectors_to_upsert,
    namespace=namespace  # Optional: add namespace if needed
)   



        
 