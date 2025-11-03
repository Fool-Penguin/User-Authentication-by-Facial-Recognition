from deepface import DeepFace
import json
import pandas as pd
import re

img_in = "input/input/realPeem3.jpg"
img_db = "faceDB"

def recognize_face(input_, database):
    try:
        # result = DeepFace.find(input_, database, normalization='ArcFace')
        result = DeepFace.find(input_, database)
        print(str(result)+"==========================")
        
        df = pd.DataFrame(result[0])
        record = json.loads(df.to_json(orient="records"))

        with open("match.json", "w") as f:
            for i in record:
                try:
                    if i['confidence'] >= 65:
                        f.write(json.dumps(i["identity"].strip("faceDB/"), indent=2))
                        f.write("\n")
                except Exception as e:
                    print("Error writing identity: ", str(e))
                    return False, None
            print("JSON File Saved")
        return True, record[0]
    
    except Exception as e:
        print("Some errors occurred: ", str(e))
        return
        
def main():
    # found, i = recognize_face(img_in, img_db)
    # name = i['identity'].strip("faceDB/").strip(".jpg") | ".png"
    found, whoami = recognize_face(img_in, img_db)
    whoami = re.sub(r'(?i)^(?:real)?\s*(peem)\d*$', r'\1', whoami["identity"].strip("faceDB\\"))
    whoami = whoami[0:len(whoami)-5]
    if(found):
        print(f"Found face is database!\n Hello nigga {(whoami)}")
    else:
        print("you're not my gng nigga😭🙏🥀")

if __name__ == "__main__":
    main()