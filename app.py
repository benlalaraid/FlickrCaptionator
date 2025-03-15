import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import random

app = FastAPI(title="Flickr Caption Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Detailed captions for different image categories
image_captions = {
    'nature': [
        "A beautiful landscape with mountains and trees in the background.",
        "A serene lake reflecting the blue sky and surrounding nature.",
        "A forest path with sunlight filtering through the trees.",
        "A majestic waterfall cascading down rocky cliffs surrounded by lush vegetation.",
        "A peaceful meadow filled with wildflowers under a clear blue sky.",
        "A stunning sunset over the mountains with vibrant orange and purple hues.",
        "A misty morning in the forest with rays of sunlight breaking through the trees.",
        "A snow-covered mountain peak against a clear blue sky.",
        "A winding river cutting through a lush green valley.",
        "A close-up of colorful autumn leaves on a forest floor."
    ],
    
    'urban': [
        "A busy city street with people walking and cars passing by.",
        "A modern skyscraper reaching into the clouds against a blue sky.",
        "A night view of a city with colorful lights illuminating the streets.",
        "An old cobblestone street with historic buildings on both sides.",
        "A bustling marketplace with vendors selling various goods.",
        "A panoramic view of a city skyline at sunset.",
        "A quiet residential neighborhood with houses lined along the street.",
        "A modern glass and steel building reflecting the surrounding cityscape.",
        "A street performer entertaining a crowd in a city square.",
        "A subway station with people waiting for the train to arrive."
    ],
    
    'people': [
        "A group of friends laughing and enjoying their time together at a park.",
        "A person deep in thought while reading a book in a cozy corner.",
        "A family having a picnic in a park on a sunny day.",
        "A street performer entertaining a crowd of onlookers with music.",
        "A child playing with toys on the floor with a big smile.",
        "A couple walking hand in hand along a beach at sunset.",
        "A group of students studying together in a library.",
        "A person jogging through a park in the early morning.",
        "A chef preparing food in a busy restaurant kitchen.",
        "A teacher explaining a lesson to attentive students in a classroom."
    ],
    
    'animals': [
        "A playful dog running through a field chasing a ball.",
        "A cat curled up sleeping in a patch of sunlight by the window.",
        "A bird with colorful plumage perched on a branch in a tree.",
        "A butterfly resting on a vibrant flower in a garden.",
        "A squirrel gathering nuts in a park during autumn.",
        "A family of ducks swimming in a pond on a sunny day.",
        "A majestic eagle soaring high in the sky with wings spread wide.",
        "A rabbit nibbling on grass in a lush green meadow.",
        "A school of colorful fish swimming in clear blue water.",
        "A horse galloping freely across an open field."
    ],
    
    'food': [
        "A delicious meal beautifully presented on a plate in a restaurant.",
        "Fresh fruits and vegetables arranged in a colorful display at a market.",
        "A steaming cup of coffee next to a pastry on a cafe table.",
        "A homemade cake decorated with berries and powdered sugar.",
        "A variety of cheese and crackers arranged on a wooden board.",
        "A bowl of hot soup with steam rising, served with fresh bread.",
        "A colorful salad with various vegetables and a drizzle of dressing.",
        "A sizzling steak being cooked on a grill with flames.",
        "A stack of pancakes topped with maple syrup and fresh berries.",
        "An assortment of sushi rolls beautifully arranged on a plate."
    ],
    
    'beach': [
        "A beautiful sandy beach with waves gently washing ashore.",
        "A person surfing on a large wave in the ocean.",
        "Children building a sandcastle on the beach during summer.",
        "Palm trees swaying in the breeze along a tropical beach.",
        "A colorful sunset over the ocean with silhouettes of boats.",
        "People playing volleyball on a sandy beach on a sunny day.",
        "A seashell collection displayed on the sand with the ocean in the background.",
        "A person relaxing in a beach chair under an umbrella by the sea.",
        "Footprints in the sand leading toward the ocean at sunrise.",
        "A lifeguard tower overlooking a busy beach during summer."
    ],
    
    'sports': [
        "A soccer player kicking a ball on a green field during a match.",
        "A basketball player dunking the ball into the hoop.",
        "A tennis player serving the ball on a court during a tournament.",
        "A group of cyclists racing along a mountain road.",
        "A swimmer diving into a pool at the start of a race.",
        "A baseball player swinging a bat at a ball during a game.",
        "A skier racing down a snow-covered mountain slope.",
        "A golfer taking a swing on a well-maintained course.",
        "A group of people playing volleyball on an indoor court.",
        "A runner crossing the finish line of a marathon with arms raised."
    ]
}

def analyze_image(image):
    """
    Analyze image properties to determine its category
    """
    try:
        # Convert image to RGB if it's not
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get image dimensions and aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        
        # Resize for faster processing
        image_small = image.resize((100, 100))
        pixels = list(image_small.getdata())
        
        # Calculate color statistics
        avg_r = sum(pixel[0] for pixel in pixels) / len(pixels)
        avg_g = sum(pixel[1] for pixel in pixels) / len(pixels)
        avg_b = sum(pixel[2] for pixel in pixels) / len(pixels)
        
        # Calculate brightness
        brightness = (avg_r + avg_g + avg_b) / 3
        
        # Calculate color variance (rough measure of colorfulness)
        r_var = sum((pixel[0] - avg_r) ** 2 for pixel in pixels) / len(pixels)
        g_var = sum((pixel[1] - avg_g) ** 2 for pixel in pixels) / len(pixels)
        b_var = sum((pixel[2] - avg_b) ** 2 for pixel in pixels) / len(pixels)
        color_variance = (r_var + g_var + b_var) / 3
        
        # Calculate edge density (rough measure of complexity)
        edge_count = 0
        for y in range(1, 99):
            for x in range(1, 99):
                # Get current pixel and neighbors
                current = image_small.getpixel((x, y))
                left = image_small.getpixel((x-1, y))
                right = image_small.getpixel((x+1, y))
                top = image_small.getpixel((x, y-1))
                bottom = image_small.getpixel((x, y+1))
                
                # Calculate differences
                diff_h = abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2])
                diff_v = abs(top[0] - bottom[0]) + abs(top[1] - bottom[1]) + abs(top[2] - bottom[2])
                
                # Count as edge if difference is significant
                if diff_h > 100 or diff_v > 100:
                    edge_count += 1
        
        edge_density = edge_count / (98 * 98)
        
        # Classify based on image properties
        # Nature: high green, low edge density
        if avg_g > max(avg_r, avg_b) and avg_g > 100 and edge_density < 0.1:
            return 'nature'
        
        # Beach: high brightness, blue dominant, low edge density
        elif brightness > 150 and avg_b > avg_r and avg_b > avg_g and edge_density < 0.1:
            return 'beach'
        
        # Urban: high edge density, moderate brightness
        elif edge_density > 0.15 and 80 < brightness < 180:
            return 'urban'
        
        # Food: high red or high color variance, square-ish aspect ratio
        elif (avg_r > max(avg_g, avg_b) or color_variance > 2000) and 0.7 < aspect_ratio < 1.3:
            return 'food'
        
        # Sports: high edge density, green or blue dominant (fields/courts/pools)
        elif edge_density > 0.12 and (avg_g > 100 or avg_b > 100):
            return 'sports'
        
        # People: moderate edge density, skin tone hints (higher red than blue)
        elif 0.08 < edge_density < 0.2 and avg_r > avg_b and 0.8 < aspect_ratio < 1.25:
            return 'people'
        
        # Animals: moderate edge density, varied colors
        elif 0.05 < edge_density < 0.15 and color_variance > 1000:
            return 'animals'
        
        # Default to a random category if no clear match
        else:
            return random.choice(list(image_captions.keys()))
            
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return random.choice(list(image_captions.keys()))

def generate_caption(image):
    """
    Generate a caption for the image based on its visual properties
    """
    try:
        # Analyze the image to determine its category
        category = analyze_image(image)
        
        # Select a caption from the appropriate category
        caption = random.choice(image_captions[category])
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        # Fallback to a generic caption
        return "A beautiful image with interesting visual elements."

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate-caption/")
async def create_caption(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate caption
        caption = generate_caption(image)
        
        return JSONResponse(content={"caption": caption, "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
