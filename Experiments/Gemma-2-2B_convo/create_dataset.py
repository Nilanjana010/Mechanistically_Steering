import os
import json
import pickle

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
corpus_path = "conversations-gone-awry-corpus"

conversations_path = os.path.join(
    corpus_path,
    "conversations.json"
)

utterances_path = os.path.join(
    corpus_path,
    "utterances.jsonl"
)


# -------------------------------------------------------
# Load conversations
# -------------------------------------------------------
with open(
    conversations_path,
    "r",
    encoding="utf-8"
) as f:
    conversations_raw = json.load(f)


# -------------------------------------------------------
# Load utterances
# -------------------------------------------------------
with open(
    utterances_path,
    "r",
    encoding="utf-8"
) as f:

    utterances_raw = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


print("Total conversations =", len(conversations_raw))
print("Total utterances =", len(utterances_raw))


# -------------------------------------------------------
# Group utterances by conversation ID
# -------------------------------------------------------
utterances_by_conversation = {}

for utterance in utterances_raw:

    conversation_id = utterance["conversation_id"]

    if conversation_id not in utterances_by_conversation:
        utterances_by_conversation[conversation_id] = []

    utterances_by_conversation[
        conversation_id
    ].append(utterance)


# -------------------------------------------------------
# Get actual conversation
# Remove section headers and sort by time
# -------------------------------------------------------
def get_conversation(conversation_id):

    utterances = utterances_by_conversation.get(
        conversation_id,
        []
    )

    # Remove Wikipedia section headings
    utterances = [
        u
        for u in utterances
        if not u["meta"].get(
            "is_section_header",
            False
        )
    ]

    # Put comments in chronological order
    utterances = sorted(
        utterances,
        key=lambda u: u["timestamp"]
    )

    return utterances


# -------------------------------------------------------
# Convert multiple utterances into one text
# -------------------------------------------------------
def make_text(utterances):

    return "\n".join(
        f'{u["speaker"]}: {u["text"].strip()}'
        for u in utterances
    )


# -------------------------------------------------------
# Select 265 personal-attack conversations
# -------------------------------------------------------
rows = []
ct = 0

for conversation_id, info in conversations_raw.items():

    # Use personal-attack conversations only
    if info["conversation_has_personal_attack"] is False:
        continue

    # keep only test conversations
    if info["split"] != "test":
        continue

    pair_id = info["pair_id"]

    # Pair must exist
    if pair_id not in conversations_raw:
        continue

    # Pair should be the civil conversation
    if (
        conversations_raw[pair_id][
            "conversation_has_personal_attack"
        ]
        is True
    ):
        continue


    # ---------------------------------------------------
    # Get attack conversation
    # ---------------------------------------------------
    conversation = get_conversation(
        conversation_id
    )

    if len(conversation) == 0:
        continue


    # ---------------------------------------------------
    # Find FIRST personal attack
    # ---------------------------------------------------
    first_attack_index = None

    for i, utterance in enumerate(conversation):

        if utterance["meta"].get(
            "comment_has_personal_attack",
            False
        ):
            first_attack_index = i
            break


    # No attack utterance found
    if first_attack_index is None:
        continue


    # We need conversation before the attack
    if first_attack_index == 0:
        continue


    # ---------------------------------------------------
    # Prompt = everything BEFORE first attack
    # ---------------------------------------------------
    prompt_utterances = conversation[
        :first_attack_index
    ]

    prompt = make_text(
        prompt_utterances
    )


    # ---------------------------------------------------
    # Response = FIRST personal attack
    # ---------------------------------------------------
    response = (
        f'{conversation[first_attack_index]["speaker"]}: '
        f'{conversation[first_attack_index]["text"].strip()}'
    )

    # ---------------------------------------------------
    # Complete attack conversation text
    # ---------------------------------------------------
    conversation_text = make_text(
        conversation
    )


    # ---------------------------------------------------
    # Get paired civil conversation
    # ---------------------------------------------------
    pair_conversation = get_conversation(
        pair_id
    )

    if len(pair_conversation) == 0:
        continue


    pair_text = make_text(
        pair_conversation
    )


    # ---------------------------------------------------
    # Keep ONLY required information
    # ---------------------------------------------------
    row = {

        "conversation_id":
            conversation_id,

        "prompt":
            prompt,

        "response":
            response,

        "conversation_text":
            conversation_text,

        "pair_id":
            pair_id,

        "pair_text":
            pair_text
    }


    rows.append(row)

    ct += 1

    if ct >= 265:
        break

output_dir = "dataset"

os.makedirs(
    output_dir,
    exist_ok=True
)


output_file = os.path.join(
    output_dir,
    "dataset_personal_attack.pkl"
)


with open(
    output_file,
    "wb"
) as f:

    pickle.dump(
        rows,
        f
    )


print("Done")
print("Number of samples =", len(rows))
print("Saved at =", output_file)