# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import subprocess
# import os
# import tempfile
# import logging
# import shutil
# import uuid
# import re
# import glob
# from moviepy import VideoFileClip, concatenate_videoclips
# import sys
# import traceback

# # Configure logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Manim Rendering Service")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create output directory
# os.makedirs("output", exist_ok=True)

# class ManimRequest(BaseModel):
#     code: str
#     scene_name: str = ""  # Empty string means render all scenes

# class ManimResponse(BaseModel):
#     success: bool
#     message: str
#     video_id: str = None
#     scenes_rendered: list = []

# def extract_scene_classes(code):
#     """Extract all Scene classes from the provided code"""
#     scene_classes = []
#     # Look for class definitions that inherit from Scene
#     pattern = r'class\s+(\w+)\s*\(\s*Scene\s*\)'
#     matches = re.finditer(pattern, code)
    
#     for match in matches:
#         scene_classes.append(match.group(1))
    
#     return scene_classes

# def combine_videos(video_files, output_path):
#     """Combine multiple video files into one"""
#     if not video_files:
#         return False
    
#     if len(video_files) == 1:
#         shutil.copy(video_files[0], output_path)
#         return True
    
#     try:
#         clips = []
#         for vf in video_files:
#             try:
#                 clip = VideoFileClip(vf)
#                 clips.append(clip)
#             except Exception as e:
#                 logger.error(f"Error loading video file {vf}: {str(e)}")
#                 continue
        
#         if not clips:
#             logger.error("No valid video clips to combine")
#             return False
        
#         if len(clips) == 1:
#             # If only one clip loaded successfully, just write it directly
#             clips[0].write_videofile(output_path, codec="libx264")
#             clips[0].close()
#             return True
            
#         final_clip = concatenate_videoclips(clips, method="compose")
#         final_clip.write_videofile(output_path, codec="libx264")
#         final_clip.close()
#         for clip in clips:
#             clip.close()
#         return True
#     except Exception as e:
#         logger.error(f"Error combining videos: {str(e)}")
#         return False

# def create_runner_script(scene_name, scene_file_path, output_dir):
#     """
#     Create a standalone Python script that will render the Manim scene
#     """
#     # Get the module name from the file path
#     module_name = os.path.splitext(os.path.basename(scene_file_path))[0]
    
#     # Create the runner script path
#     runner_path = os.path.join(os.path.dirname(scene_file_path), f"runner_{uuid.uuid4()}.py")
    
#     # Fix paths for f-string (replace backslashes with forward slashes)
#     safe_output_dir = output_dir.replace("\\", "/")
#     safe_video_dir = os.path.join(output_dir, "videos").replace("\\", "/")
#     safe_scene_dir = os.path.dirname(scene_file_path).replace("\\", "/")
    
#     # Write the runner script
#     with open(runner_path, 'w', encoding='utf-8') as f:
#         f.write(f'''
# import os
# import sys

# # Set the PYTHONPATH to find the Manim module in the scene file
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# # Import manim
# from manim import *

# # Import the scene module
# sys.path.insert(0, "{safe_scene_dir}")
# imported_module = __import__('{module_name}')

# # Configure manim settings
# config.media_dir = "{safe_output_dir}"
# config.video_dir = "{safe_video_dir}"
# config.output_file = "video"
# config.frame_rate = 30
# config.pixel_height = 1080
# config.pixel_width = 1920
# config.preview = False

# # Create and render the scene
# scene = imported_module.{scene_name}()
# scene.render()

# print("MANIM_RENDER_COMPLETE: Scene rendered successfully!")
# ''')
    
#     return runner_path

# @app.post("/render", response_model=ManimResponse)
# async def render_manim(request: ManimRequest):
#     """
#     Render Manim code with improved direct Python script approach
#     """
#     video_id = str(uuid.uuid4())
#     temp_dir = None
    
#     try:
#         # Create a unique temporary directory
#         temp_dir = tempfile.mkdtemp(prefix=f"manim_{video_id}_")
#         logger.info(f"Created temp directory: {temp_dir}")
        
#         # Create media directories
#         media_dir = os.path.join(temp_dir, "media")
#         video_dir = os.path.join(media_dir, "videos")
#         os.makedirs(video_dir, exist_ok=True)
        
#         output_file = f"output/video_{video_id}.mp4"
        
#         # Create a uniquely named Python file for the scene code
#         scene_file = os.path.join(temp_dir, f"scene_{video_id}.py")
        
#         # Ensure imports are at the top by adding them if missing
#         code_to_write = request.code
#         if "from manim import" not in code_to_write and "import manim" not in code_to_write:
#             code_to_write = "from manim import *\n\n" + code_to_write
            
#         if "import numpy as np" not in code_to_write and "np." in code_to_write:
#             code_to_write = "import numpy as np\n" + code_to_write
        
#         # Write the scene code to file
#         with open(scene_file, "w", encoding="utf-8") as f:
#             f.write(code_to_write)
        
#         logger.info(f"Saved scene code to: {scene_file}")
        
#         # Determine scene name
#         scenes_to_render = []
#         if request.scene_name:
#             scenes_to_render = [request.scene_name]
#         else:
#             # Look specifically for MainScene first
#             if "class MainScene(Scene)" in code_to_write:
#                 scenes_to_render = ["MainScene"]
#             else:
#                 # Fall back to extracting all scenes
#                 scenes_to_render = extract_scene_classes(code_to_write)
        
#         if not scenes_to_render:
#             return ManimResponse(
#                 success=False, 
#                 message="No Scene classes found in the provided code."
#             )
        
#         logger.info(f"Scenes to render: {scenes_to_render}")
        
#         # Track rendered video files
#         rendered_videos = []
#         scenes_rendered = []
        
#         # Render each scene
#         for scene_name in scenes_to_render:
#             logger.info(f"Creating runner script for scene: {scene_name}")
            
#             # Create a dedicated runner script for this scene
#             runner_script = create_runner_script(scene_name, scene_file, media_dir)
            
#             try:
#                 # Run the runner script as a separate process
#                 logger.info(f"Running scene {scene_name} with Python")
                
#                 # Get the original directory
#                 original_dir = os.getcwd()
                
#                 # Change to the temp directory to run the script
#                 os.chdir(temp_dir)
                
#                 # Use Python to run the runner script
#                 result = subprocess.run(
#                     [sys.executable, runner_script],
#                     capture_output=True,
#                     text=True,
#                     encoding='utf-8',
#                     errors='replace'  # Handle non-UTF-8 characters
#                 )
                
#                 # Change back to the original directory
#                 os.chdir(original_dir)
                
#                 # Check if rendering was successful
#                 if result.returncode != 0:
#                     logger.error(f"Failed to render scene {scene_name}")
#                     logger.error(f"Error output: {result.stderr}")
#                     continue
                
#                 logger.info(f"Runner script output: {result.stdout[:500]}")
                
#                 # Find the rendered video file
#                 rendered_file_found = False
                
#                 # Search in all possible locations
#                 search_dirs = [
#                     os.path.join(media_dir, "videos"),
#                     os.path.join(media_dir, "videos", f"scene_{video_id}"),
#                     os.path.join(media_dir, "videos", f"scene_{video_id}", "1080p60"),
#                     os.path.join(media_dir, "videos", f"scene_{video_id}", "720p30"),
#                     os.path.join(media_dir, "videos", f"scene_{video_id}", "480p15")
#                 ]
                
#                 for search_dir in search_dirs:
#                     if os.path.exists(search_dir):
#                         video_files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) 
#                                       if f.endswith('.mp4') and scene_name in f]
                        
#                         if video_files:
#                             # Get the most recent video file
#                             latest_video = max(video_files, key=os.path.getmtime)
#                             rendered_videos.append(latest_video)
#                             scenes_rendered.append(scene_name)
#                             logger.info(f"Found rendered video for {scene_name}: {latest_video}")
#                             rendered_file_found = True
#                             break
                
#                 # If still not found, search the entire media directory
#                 if not rendered_file_found:
#                     logger.info("Searching entire media directory for video files...")
#                     for root, dirs, files in os.walk(media_dir):
#                         for file in files:
#                             if file.endswith('.mp4'):
#                                 video_path = os.path.join(root, file)
#                                 # If we have multiple scenes, check if the file name contains the scene name
#                                 if len(scenes_to_render) > 1 and scene_name not in file:
#                                     continue
                                
#                                 rendered_videos.append(video_path)
#                                 scenes_rendered.append(scene_name)
#                                 logger.info(f"Found video in broader search: {video_path}")
#                                 rendered_file_found = True
#                                 break
#                         if rendered_file_found:
#                             break
                
#                 if not rendered_file_found:
#                     logger.warning(f"No video file found for scene {scene_name}")
                
#             except Exception as e:
#                 logger.exception(f"Error running runner script for {scene_name}: {str(e)}")
#             finally:
#                 # Clean up the runner script
#                 try:
#                     if os.path.exists(runner_script):
#                         os.remove(runner_script)
#                 except Exception as e:
#                     logger.warning(f"Failed to clean up runner script: {str(e)}")
        
#         if not rendered_videos:
#             return ManimResponse(
#                 success=False, 
#                 message="No videos were successfully rendered. Check logs for details."
#             )
        
#         # Combine videos or copy the single video to output location
#         if len(rendered_videos) == 1:
#             # If only one video, simply copy it to the output location
#             shutil.copy(rendered_videos[0], output_file)
#             logger.info(f"Single video copied to: {output_file}")
#             success = True
#         else:
#             # For multiple videos, use the combine function
#             success = combine_videos(rendered_videos, output_file)
#             logger.info(f"Multiple videos combined to: {output_file}")
        
#         if success:
#             return ManimResponse(
#                 success=True, 
#                 message=f"Successfully rendered {len(scenes_rendered)} scenes.",
#                 video_id=video_id,
#                 scenes_rendered=scenes_rendered
#             )
#         else:
#             return ManimResponse(
#                 success=False, 
#                 message="Failed to combine rendered videos."
#             )
    
#     except Exception as e:
#         logger.exception(f"Error in render endpoint: {str(e)}")
#         return ManimResponse(success=False, message=f"Error: {str(e)}")
    
#     finally:
#         # Clean up temp directory
#         if temp_dir and os.path.exists(temp_dir):
#             try:
#                 shutil.rmtree(temp_dir)
#                 logger.info(f"Cleaned up temp directory: {temp_dir}")
#             except Exception as e:
#                 logger.warning(f"Failed to clean up temp directory: {str(e)}")

# @app.get("/video/{video_id}")
# async def get_video(video_id: str):
#     """
#     Get a rendered video by ID
#     """
#     video_path = f"output/video_{video_id}.mp4"
#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="Video not found")
    
#     return FileResponse(
#         path=video_path,
#         media_type="video/mp4",
#         filename=f"manim_animation_{video_id}.mp4"
#     )

# @app.get("/")
# async def root():
#     """
#     Root endpoint
#     """
#     return {"message": "Manim Rendering Service is running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import tempfile
import logging
import shutil
import uuid
import re
import glob
import signal
import time
from moviepy import VideoFileClip, concatenate_videoclips
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app")  # Changed to "app" for consistent logging

app = FastAPI(title="Manim Rendering Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
os.makedirs("output", exist_ok=True)

# Store rendered files by topic for caching
CACHE_DIR = os.path.join("output", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class ManimRequest(BaseModel):
    code: str
    scene_name: str = ""  # Empty string means render all scenes
    topic: str = ""  # Added for caching purposes

class ManimResponse(BaseModel):
    success: bool
    message: str
    video_id: str = None
    scenes_rendered: list = []

def extract_scene_classes(code):
    """Extract all Scene classes from the provided code"""
    scene_classes = []
    pattern = r'class\s+(\w+)\s*\(\s*Scene\s*\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        scene_classes.append(match.group(1))
    
    return scene_classes

def combine_videos(video_files, output_path):
    """Combine multiple video files into one"""
    if not video_files:
        return False
    
    if len(video_files) == 1:
        shutil.copy(video_files[0], output_path)
        return True
    
    try:
        clips = []
        for vf in video_files:
            try:
                clip = VideoFileClip(vf)
                clips.append(clip)
            except Exception as e:
                logger.error(f"Error loading video file {vf}: {str(e)}")
                continue
        
        if not clips:
            logger.error("No valid video clips to combine")
            return False
        
        if len(clips) == 1:
            clips[0].write_videofile(output_path, codec="libx264")
            clips[0].close()
            return True
            
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_path, codec="libx264")
        final_clip.close()
        for clip in clips:
            clip.close()
        return True
    except Exception as e:
        logger.error(f"Error combining videos: {str(e)}")
        return False

def create_runner_script(scene_name, scene_file_path, output_dir):
    """
    Create a low-quality, optimized runner script for deployment environments
    """
    module_name = os.path.splitext(os.path.basename(scene_file_path))[0]
    runner_path = os.path.join(os.path.dirname(scene_file_path), f"runner_{uuid.uuid4()}.py")
    
    safe_output_dir = output_dir.replace("\\", "/")
    safe_video_dir = os.path.join(output_dir, "videos").replace("\\", "/")
    safe_scene_dir = os.path.dirname(scene_file_path).replace("\\", "/")
    
    # Create a script with low-quality settings to reduce resource usage
    with open(runner_path, 'w', encoding='utf-8') as f:
        f.write(f'''
import os
import sys
import signal

# Set timeout handler to prevent hanging
def timeout_handler(signum, frame):
    print("Rendering timed out!")
    sys.exit(1)

# Set a 120-second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(120)

# Set the PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import manim with optimized settings
from manim import *

# Import the scene module
sys.path.insert(0, "{safe_scene_dir}")
imported_module = __import__('{module_name}')

# Configure manim for low resource usage
config.media_dir = "{safe_output_dir}"
config.video_dir = "{safe_video_dir}"
config.output_file = "video"
config.frame_rate = 15  # Lower frame rate
config.pixel_height = 480  # Lower resolution
config.pixel_width = 854  # Lower resolution
config.preview = False
config.disable_caching = True  # Disable caching to save memory
config.verbosity = "ERROR"  # Reduce logging

# Create and render the scene
try:
    # Get the scene class
    scene = imported_module.{scene_name}()
    
    # Render with optimized settings
    scene.render(preview=False)
    print("MANIM_RENDER_COMPLETE: Scene rendered successfully!")
except Exception as e:
    print(f"Rendering error: {{str(e)}}")
    sys.exit(1)
finally:
    # Reset the alarm
    signal.alarm(0)
''')
    
    return runner_path

def check_cache(topic, scene_name):
    """Check if we have a cached video for this topic and scene"""
    if not topic:
        return None
    
    # Create a cache key
    cache_key = re.sub(r'[^a-zA-Z0-9]', '_', topic.lower())
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}_{scene_name}.mp4")
    
    if os.path.exists(cache_path):
        cache_id = os.path.basename(cache_path).split('.')[0]
        return cache_id
    
    return None

def save_to_cache(topic, scene_name, video_path):
    """Save a video to the cache"""
    if not topic:
        return
    
    # Create a cache key
    cache_key = re.sub(r'[^a-zA-Z0-9]', '_', topic.lower())
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}_{scene_name}.mp4")
    
    try:
        shutil.copy(video_path, cache_path)
        logger.info(f"Saved video to cache: {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save to cache: {str(e)}")

@app.post("/render", response_model=ManimResponse)
async def render_manim(request: ManimRequest):
    """
    Render Manim code with improved resilience for deployment
    """
    # Check cache first if topic is provided
    if request.topic:
        cache_video_id = check_cache(request.topic, request.scene_name or "MainScene")
        if cache_video_id:
            logger.info(f"Found cached video for topic: {request.topic}")
            return ManimResponse(
                success=True,
                message="Retrieved from cache",
                video_id=cache_video_id,
                scenes_rendered=[request.scene_name or "MainScene"]
            )
    
    video_id = str(uuid.uuid4())
    temp_dir = None
    
    try:
        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"manim_{video_id}_")
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Create media directories
        media_dir = os.path.join(temp_dir, "media")
        video_dir = os.path.join(media_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        output_file = f"output/video_{video_id}.mp4"
        
        # Create a uniquely named Python file for the scene code
        scene_file = os.path.join(temp_dir, f"scene_{video_id}.py")
        
        # Ensure imports are at the top
        code_to_write = request.code
        if "from manim import" not in code_to_write and "import manim" not in code_to_write:
            code_to_write = "from manim import *\n\n" + code_to_write
            
        if "import numpy as np" not in code_to_write and "np." in code_to_write:
            code_to_write = "import numpy as np\n" + code_to_write
        
        # Write the scene code to file
        with open(scene_file, "w", encoding="utf-8") as f:
            f.write(code_to_write)
        
        logger.info(f"Saved scene code to: {scene_file}")
        
        # Determine scene name
        scenes_to_render = []
        if request.scene_name:
            scenes_to_render = [request.scene_name]
        else:
            # Look specifically for MainScene first
            if "class MainScene(Scene)" in code_to_write:
                scenes_to_render = ["MainScene"]
            else:
                # Fall back to extracting all scenes
                scenes_to_render = extract_scene_classes(code_to_write)
        
        if not scenes_to_render:
            return ManimResponse(
                success=False, 
                message="No Scene classes found in the provided code."
            )
        
        logger.info(f"Scenes to render: {scenes_to_render}")
        
        # Track rendered video files
        rendered_videos = []
        scenes_rendered = []
        
        # Render each scene
        for scene_name in scenes_to_render:
            logger.info(f"Creating runner script for scene: {scene_name}")
            
            # Create a dedicated runner script for this scene
            runner_script = create_runner_script(scene_name, scene_file, media_dir)
            
            try:
                # Run the runner script as a separate process
                logger.info(f"Running scene {scene_name} with Python")
                
                # Get the original directory
                original_dir = os.getcwd()
                
                # Change to the temp directory to run the script
                os.chdir(temp_dir)
                
                # Run with a timeout to prevent hanging
                process = subprocess.Popen(
                    [sys.executable, runner_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # Wait with timeout (3 minutes)
                try:
                    stdout, stderr = process.communicate(timeout=180)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    # Kill the process if it times out
                    process.kill()
                    stdout, stderr = process.communicate()
                    returncode = -1
                    logger.error("Rendering process timed out after 180 seconds")
                
                # Change back to the original directory
                os.chdir(original_dir)
                
                if returncode != 0:
                    logger.error(f"Failed to render scene {scene_name}")
                    logger.error(f"Error output: {stderr}")
                    # Continue to the next scene - don't abort
                    continue
                
                # Find the rendered video file
                rendered_file_found = False
                
                # Search for video files
                for root, dirs, files in os.walk(media_dir):
                    for file in files:
                        if file.endswith('.mp4'):
                            video_path = os.path.join(root, file)
                            rendered_videos.append(video_path)
                            scenes_rendered.append(scene_name)
                            logger.info(f"Found video: {video_path}")
                            rendered_file_found = True
                            break
                    if rendered_file_found:
                        break
                
                if not rendered_file_found:
                    logger.warning(f"No video file found for scene {scene_name}")
                
            except Exception as e:
                logger.exception(f"Error running runner script for {scene_name}: {str(e)}")
            finally:
                # Clean up the runner script
                try:
                    if os.path.exists(runner_script):
                        os.remove(runner_script)
                except:
                    pass
        
        if not rendered_videos:
            # If all rendering attempts failed, provide a more helpful error message
            return ManimResponse(
                success=False, 
                message="Rendering failed. The animation may be too complex for our current resources."
            )
        
        # Combine videos or copy the single video to output location
        if len(rendered_videos) == 1:
            # If only one video, simply copy it to the output location
            shutil.copy(rendered_videos[0], output_file)
            logger.info(f"Single video copied to: {output_file}")
            success = True
        else:
            # For multiple videos, use the combine function
            success = combine_videos(rendered_videos, output_file)
            logger.info(f"Multiple videos combined to: {output_file}")
        
        if success:
            # Save to cache if a topic was provided
            if request.topic and os.path.exists(output_file):
                save_to_cache(request.topic, scenes_to_render[0], output_file)
                
            return ManimResponse(
                success=True, 
                message=f"Successfully rendered {len(scenes_rendered)} scenes.",
                video_id=video_id,
                scenes_rendered=scenes_rendered
            )
        else:
            return ManimResponse(
                success=False, 
                message="Failed to combine rendered videos."
            )
    
    except Exception as e:
        logger.exception(f"Error in render endpoint: {str(e)}")
        return ManimResponse(success=False, message=f"Error: {str(e)}")
    
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {str(e)}")

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """
    Get a rendered video by ID
    """
    # Check both the main output directory and cache directory
    video_path = f"output/video_{video_id}.mp4"
    cache_path = None
    
    # If not found in output, check the cache
    if not os.path.exists(video_path):
        for file in os.listdir(CACHE_DIR):
            if file.startswith(video_id) and file.endswith(".mp4"):
                cache_path = os.path.join(CACHE_DIR, file)
                break
    
    final_path = video_path if os.path.exists(video_path) else cache_path
    
    if not final_path or not os.path.exists(final_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=final_path,
        media_type="video/mp4",
        filename=f"math_animation_{video_id}.mp4"
    )

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {"message": "Manim Rendering Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
