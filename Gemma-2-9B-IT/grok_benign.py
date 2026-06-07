import pickle, os
from xai_sdk import Client
from xai_sdk.chat import user, system

folder = "dataset"
all_data = None
dict_new = {}
client = Client(
                        api_key="****",
                        timeout=3600, # Override default timeout with longer timeout for reasoning models
                    )
chat = client.chat.create(model="grok-4-1-fast-non-reasoning")

for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            all_data = pickle.load(f)
        ct = 0    
        if filename not in dict_new:
             dict_new[filename] = {}
        for i in range(len(all_data)):
            ct += 1
            user_pr = all_data[i]['prompt']
            print("user_pr = ", user_pr)
            input_text = "Create a benign looking version of the prompt:'" + user_pr + "'. Provide your output after 'Answer is:'. Do not add anything to the response beyond what is asked."     
            chat.append(system("You are a helpful assistant."))
            chat.append(user(input_text))
            response = chat.sample()
            content_grok = response.content
            print("response.content default == ", content_grok)
            if i not in dict_new[filename]:
                 dict_new[filename][i] = {}
            dict_new[filename][i]['original'] = user_pr
            dict_new[filename][i]['new'] = content_grok
        print(ct)      

with open("benign_prompts.pkl", "wb") as f:
    pickle.dump(dict_new, f) 


