import face_recognition
import os
from tqdm import tqdm
from typing import Any
import numpy.typing as npt
import pickle
import cv2
import cv2.typing as cvt


UNKNOWN_DIR_NAME = "all-unknown"

NUM_TIME_TO_UPSAMPLE: int = 1
CHOSEN_MODEL = "hog"  # "cnn" or "hog" (cnn more accurate but slower)
FREQUENT_FACE_THRESHOLD = 20


def recognize(image_path: str):
    face_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(face_image, NUM_TIME_TO_UPSAMPLE, CHOSEN_MODEL)
    encodings = face_recognition.face_encodings(face_image, face_locations)
    num_face: int = len(encodings)  # type: ignore
    return encodings, num_face, face_locations


class ImgFace:
    def __init__(self, img_path: str, img_name: str, encoding: npt.NDArray[Any], face_location: npt.NDArray[Any]):
        self.img_path = img_path
        self.img_name = img_name
        self.encoding = encoding
        self.face_location = face_location


def main():
    cache_encoding_dict: dict[str, list[ImgFace]] = {}

    # Store Result
    no_face_imgs: list[str] = []
    single_face_imgs: list[str] = []
    multi_face_imgs: list[str] = []
    frequent_face_classes: dict[str, list[str]] = {}

    # ============== Loop all imgs from unknown folder ==============
    for img_name in tqdm(os.listdir(UNKNOWN_DIR_NAME)):
        image_path = f"{UNKNOWN_DIR_NAME}/{img_name}"
        encodings, num_face, face_locations = recognize(image_path)

        # Condition1: no face detected
        if not (num_face >= 1):
            no_face_imgs.append(image_path)
            continue

        if num_face == 1:
            single_face_imgs.append(image_path)
        if num_face > 1:
            multi_face_imgs.append(image_path)

        # any face detected
        # Loop all faces in one image
        for i in range(num_face):
            currImgFace = ImgFace(image_path, img_name, encodings[i], face_locations[i])  # type: ignore
            any_matched = False

            for i in range(len(cache_encoding_dict)):
                known_ec = [item.encoding for item in cache_encoding_dict[str(i)]]
                results = face_recognition.compare_faces(known_ec, currImgFace.encoding)

                # Condition2: if this face match current known encoding, then add this encoding
                if all(r == True for r in results):
                    cache_encoding_dict[str(i)] = [*cache_encoding_dict[str(i)], currImgFace]
                    any_matched = True
                    break

            # Condition3: no match any in dict, then create one new category
            if any_matched == False:
                cache_encoding_dict[str(len(cache_encoding_dict))] = [currImgFace]


    print("Classes:")
    for i in range(len(cache_encoding_dict)):
        curr_dict = cache_encoding_dict[str(i)]
        num = len(curr_dict)
        if num >= 1:
            print(i, [item.img_path for item in curr_dict])
            print("===============")

    # ============================= Post-process =============================
    # ============== Crop image with class ==============
    for key, value in cache_encoding_dict.items():
        if not len(value) > FREQUENT_FACE_THRESHOLD:
            continue

        frequent_face_classes[key] = [v.img_path for v in value]

        dir_path = f"out/{key}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # select two image in class and crop it
        for box_i in range(2):
            box_data = value[box_i]
            img_path = box_data.img_path
            print(img_path)

            # Read image
            img = cv2.imread(img_path)

            # Crop image with face location
            top, right, bottom, left = box_data.face_location
            face_img: cvt.MatLike = img[top:bottom, left:right]

            # Copy cropped image to a new file
            # create folder if not exist

            new_file_path = f"{dir_path}/cropped_class-{key}_{box_i}-{box_data.img_name}"
            cv2.imwrite(new_file_path, face_img)

            print(f"Cropped image saved: {new_file_path}")


    # Save Face Class Encodings
    with open("out/encoding-dict.pkl", "wb+") as fp:
        pickle.dump(cache_encoding_dict, fp)
        print(f"Saved encoding dictionary: out/encoding-dict.pkl")


    if not os.path.exists("out/results"):
            os.makedirs("out/results")
    # Save All Result
    with open("out/results/no-face.pkl", "wb") as fp:
        pickle.dump(no_face_imgs, fp)
        print(f"Saved no face images: out/results/no-face.pkl")

    with open("out/results/single-face.pkl", "wb") as fp:
        pickle.dump(single_face_imgs, fp)
        print(f"Saved single face images: out/results/single-face.pkl")
        
    with open("out/results/multi-face.pkl", "wb") as fp:
        pickle.dump(multi_face_imgs, fp)
        print(f"Saved multi face images: out/results/multi-face.pkl")
        
    with open('out/results/frequent-face-classes.pkl','wb') as fp:
        pickle.dump(frequent_face_classes, fp)
        print(f"Saved frequent face classes: out/results/frequent-face-classes.pkl")


if __name__ == "__main__":
    main()
