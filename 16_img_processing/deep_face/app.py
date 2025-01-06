from deepface import DeepFace

result = DeepFace.verify(
  img1_path = "img1.png",
  img2_path = "img2.png",
)

print(result)


embedding_objs = DeepFace.represent(
  img_path = "img1.png"
)

dfs = DeepFace.find(
  img_path = "img1.png",
  db_path = "images",
)


img_path = "img1.png"
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
embedding = DeepFace.represent(img_path, model_name = models[0])
print("embedding:",embedding)

print("--------")

print("--------")

print("--------")

print(embedding_objs)


# for embedding_obj in embedding_objs:
#   embedding = embedding_obj["embedding"]
#   assert isinstance(embedding, list)
#   assert (
#     model_name == "VGG-Face"
#     and len(embedding) == 4096
#   )


models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet"
]

#face verification
result = DeepFace.verify(
  img1_path = "img1.png",
  img2_path = "img2.png",
  model_name = models[0],
)

print("again:",result)

#face recognition
dfs = DeepFace.find(
  img_path = "img1.png",
  db_path = "images", 
  model_name = models[1],
)

print("dfs:", dfs)

#embeddings
embedding_objs = DeepFace.represent(
  img_path = "img1.png",
  model_name = models[2],
)


print("embedding_objs:", embedding_objs)






# import cv2

# import sqlite3



# # Function to extract facial features (replace with your deep learning model)

# def extract_face_features(image):

#     # ... your feature extraction logic here

#     return features



# def create_database(db_name):

#     conn = sqlite3.connect(db_name)

#     cursor = conn.cursor()

#     cursor.execute('''CREATE TABLE IF NOT EXISTS users 

#                     (user_id INTEGER PRIMARY KEY, facial_embedding BLOB)''')

#     conn.commit()

#     conn.close()



# # Example usage

# image = cv2.imread("img1.png")

# features = extract_face_features(image)

# create_database("face_db.db")



# conn = sqlite3.connect("face_db.db")

# cursor = conn.cursor()

# cursor.execute("INSERT INTO users (facial_embedding) VALUES (?)", (features,)) 

# conn.commit()

# conn.close() 
