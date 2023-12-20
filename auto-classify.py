import face_recognition
import os
from tqdm import tqdm
from typing import Any
import numpy.typing as npt
import pickle


def recognize(image_path: str):
    face_image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(face_image)
    num_face: int = len(encodings)  # type: ignore
    face_locations = face_recognition.face_locations(face_image)
    return encodings, num_face, face_locations


class ImgFace:
    def __init__(self, img_path: str, encoding: npt.NDArray[Any], face_location: npt.NDArray[Any]):
        self.img_path = img_path
        self.encoding = encoding
        self.face_location = face_location


# TODO: get boundbox of face in encoding
# TODO: label num_face


def main():
    unknown_dir_name = "./all-unknown"

    # key(index): (img_path, encodings)[]
    cache_encoding_dict: dict[str, list[ImgFace]] = {}
    no_face_imgs = []

    # Loop all imgs from unknown folder
    for image_name in tqdm(os.listdir(unknown_dir_name)):
        image_path = f"{unknown_dir_name}/{image_name}"
        encodings, num_face, face_locations = recognize(image_path)

        # Condition1: no face detected
        if not (num_face >= 1):
            no_face_imgs.append(image_path)
            continue

        # any face detected
        # Loop all faces in one image
        for i in range(num_face):
            currImgFace =  ImgFace(image_path, encodings[i], face_locations[i])
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

    with open("out/encoding-dict.pkl", "wb+") as fp:
        pickle.dump(cache_encoding_dict, fp)
        print(f"Saved encoding dictionary: out/encoding-dict.pkl")

    print("Classes:")
    for i in range(len(cache_encoding_dict)):
        curr_dict = cache_encoding_dict[str(i)]
        num = len(curr_dict)
        if num >= 1:
            print(i, [item.img_path for item in curr_dict])
            print("===============")


if __name__ == "__main__":
    main()
    # data: dict[str, list[ImgFace]] = pickle.load(open("out/encoding-dict.pkl", "rb"))
    
