import os
import pickle
from xai_sdk import Client
from xai_sdk.chat import user, system

folder = "dataset"
all_data = None

concept_Folder = "concept_dictionaries"
os.makedirs(concept_Folder, exist_ok=True)

for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            all_data = pickle.load(f)

        list_concepts = {}
        for i in range(len(all_data)):    
            if all_data[i]['is_safe'] == False: 
                print("i is == ", i)  
                print("\n\n\n\nPrompt:", all_data[i]['prompt'], "\n\n\n\n")    
                input_text = "Extract words directly from the text that focus on negative sentiments. Output the words after 'Answer is:' as comma separated. Here is the text:-" + all_data[i]['response']
                print("input_text is = ", input_text)

                client = Client(
                        api_key="****",
                        timeout=3600, # Override default timeout with longer timeout for reasoning models
                    )

                chat = client.chat.create(model="grok-4-1-fast-non-reasoning")
                                                
                chat.append(system("You are a helpful assistant."))
                chat.append(user(input_text))

                response = chat.sample()
                print("response.content default == ", response.content)

                list_concepts[i] = [response.content]

        with open(os.path.join(concept_Folder, "concept_dictionary_"+ filename), "wb") as f:
            pickle.dump(list_concepts, f)