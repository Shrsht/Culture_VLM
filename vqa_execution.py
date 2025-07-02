# --- VQA Task Specific Functions ---
import PIL
from PIL import Image

def get_image_from_hf_dataset(dataset, index):
    """Retrieves a PIL Image from a Hugging Face dataset."""
    try:
        return dataset['train'][index]['image']
    except (KeyError, IndexError) as e:
        print(f"Error retrieving image at index {index}: {e}")
        return None

def create_vqa_prompt(cultural_image: PIL.Image.Image, question_type: str, options_dict: dict, correct_answer_flags: list or str) -> str:
    """
      Creates a VQA prompt for the Gemini model based on the MCQ type. Returns the prompt string.
    """
    options_list_str = "\n".join([f"{letter}. {flag_name}" for flag_name, letter in options_dict.items()])

    if question_type == 'q1':
        # Q1: "Choose the flag that is from the same subregion/continent but not depicted"
        prompt = (
            "Examine the cultural context in the following image. From the list of flags below, "
            "select the flag that least represents the culture depicted in this particular image "
            "Provide your answer as only the letter of your chosen option, followed by the flag name, "
            "(e.g., 'A. Flag Name').\n\n"
            "Flags:\n"
            f"{options_list_str}\n\n"
            "Your Answer:"
        )
    elif question_type == 'q2':
        # Q2: "Choose two flags that are from the same subregion/continent but not depicted"
        prompt = (
            "Examine the cultural context in the following image. From the list of flags below, "
            "group the flags into 2 groups, one group that most represents the culture depicted in this particular image"
            "and another group, that do *not* represent the culture directly shown or implied in the image.\n"
            "Provide your answer as only the letters of your chosen options, followed by the flag names, "
            " (e.g., Group 1: 'A, B. Flag Name 1, Flag Name 2, Group 2: 'C, D. Flag Name 3, FLag Name 4').\n\n"
            "Flags:\n"
            f"{options_list_str}\n\n"
            "Your Answer:"
        )
    else:
        raise ValueError("Invalid question_type. Must be 'q1' or 'q2'.")

    return prompt

def query_gemini_vqa(cultural_image: Image.Image, prompt: str) -> str:
    """
    Sends the image and prompt to the Gemini model and returns its response.
    """
    try:
        response = model.generate_content([cultural_image, prompt])
        return response.text.strip()
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return "ERROR"

def evaluate_gemini_answer(gemini_response: str, options_dict: dict, correct_answer_info: list or str) -> bool:
    """
    Evaluates if Gemini's answer matches the correct MCQ answer.

    Args:
        gemini_response: The raw text response from Gemini.
        options_dict: The dictionary of flag names to their option letters.
        correct_answer_info: The correct answer(s) from your MCQ generation logic (letter(s)).
                             For q1, this is a single letter (e.g., 'A').
                             For q2, this is a list of letters (e.g., ['B', 'D']).
    """
    gemini_response_lower = gemini_response.lower()

    if isinstance(correct_answer_info, str): # Q1 type
        expected_letter = correct_answer_info.lower()
        # Check if the letter is present and if the corresponding flag name is also mentioned
        if expected_letter in gemini_response_lower:
            # Find the flag name corresponding to the expected letter
            correct_flag_name = next((flag for flag, letter in options_dict.items() if letter.lower() == expected_letter), None)
            if correct_flag_name and correct_flag_name.lower().replace(" ", "") in gemini_response_lower:
                return True
        return False
    elif isinstance(correct_answer_info, list): # Q2 type
        expected_letters = sorted([letter.lower() for letter in correct_answer_info])
        # Try to extract letters from Gemini's response
        found_letters = []
        for letter in "abcdefghijklmnopqrstuvwxyz":
            if letter + "." in gemini_response_lower or letter + "," in gemini_response_lower: # Account for variations
                found_letters.append(letter)
        found_letters = sorted(found_letters)

        # Check if the set of expected letters matches the set of found letters
        if set(expected_letters) == set(found_letters):
            # Additionally, check if the corresponding flag names are mentioned
            correct_flag_names = [flag for flag, letter in options_dict.items() if letter.lower() in expected_letters]
            all_flags_mentioned = True
            for flag_name in correct_flag_names:
                if flag_name.lower().replace(" ", "") not in gemini_response_lower:
                    all_flags_mentioned = False
                    break
            return all_flags_mentioned
        return False
    return False

# # --- ORIGINAL VQA Task Specific Functions ---

# def get_image_from_hf_dataset(dataset, index):
#     """Retrieves a PIL Image from a Hugging Face dataset."""
#     try:
#         return dataset['train'][index]['image']
#     except (KeyError, IndexError) as e:
#         print(f"Error retrieving image at index {index}: {e}")
#         return None

# def create_vqa_prompt(cultural_image: Image.Image, question_type: str, options_dict: dict, correct_answer_flags: list or str) -> str:
#     """
#     Creates a VQA prompt for the Gemini model based on the MCQ type.

#     Args:
#         cultural_image: PIL Image object.
#         question_type: 'q1' or 'q2'
#         options_dict: A dictionary mapping flag names to their option letters (e.g., {'India': 'A', 'Japan': 'B'})
#         correct_answer_flags: The flag name(s) that are considered 'correct' by your MCQ logic.
#                               For q1, this is a string. For q2, this is a list of strings.

#     Returns:
#         The prompt string.
#     """
#     options_list_str = "\n".join([f"{letter}. {flag_name}" for flag_name, letter in options_dict.items()])

#     if question_type == 'q1':
#         # Q1: "Choose the flag that is from the same subregion/continent but not depicted"
#         prompt = (
#             "Examine the cultural context in the following image. From the list of flags below, "
#             "select the flag that represents a country from the same general geographic region (continent or subregion) "
#             "as the culture depicted, but is *not* one of the main countries directly shown or implied in the image.\n"
#             "Provide your answer as only the letter of your chosen option, followed by the flag name, "
#             "and a brief explanation (e.g., 'A. Flag Name - Reason').\n\n"
#             "Flags:\n"
#             f"{options_list_str}\n\n"
#             "Your Answer:"
#         )
#     elif question_type == 'q2':
#         # Q2: "Choose two flags that are from the same subregion/continent but not depicted"
#         prompt = (
#             "Examine the cultural context in the following image. From the list of flags below, "
#             "select *two* flags that represent countries from the same general geographic region (continent or subregion) "
#             "as the culture depicted, but are *not* among the main countries directly shown or implied in the image.\n"
#             "Provide your answer as only the letters of your chosen options, followed by the flag names, "
#             "and a brief explanation (e.g., 'A, B. Flag Name 1, Flag Name 2 - Reason').\n\n"
#             "Flags:\n"
#             f"{options_list_str}\n\n"
#             "Your Answer:"
#         )
#     else:
#         raise ValueError("Invalid question_type. Must be 'q1' or 'q2'.")

#     return prompt

# def query_gemini_vqa(cultural_image: Image.Image, prompt: str) -> str:
#     """
#     Sends the image and prompt to the Gemini model and returns its response.
#     """
#     try:
#         response = model.generate_content([cultural_image, prompt])
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error querying Gemini: {e}")
#         return "ERROR"

# def evaluate_gemini_answer(gemini_response: str, options_dict: dict, correct_answer_info: list or str) -> bool:
#     """
#     Evaluates if Gemini's answer matches the correct MCQ answer.

#     Args:
#         gemini_response: The raw text response from Gemini.
#         options_dict: The dictionary of flag names to their option letters.
#         correct_answer_info: The correct answer(s) from your MCQ generation logic (letter(s)).
#                              For q1, this is a single letter (e.g., 'A').
#                              For q2, this is a list of letters (e.g., ['B', 'D']).
#     """
#     gemini_response_lower = gemini_response.lower()

#     if isinstance(correct_answer_info, str): # Q1 type
#         expected_letter = correct_answer_info.lower()
#         # Check if the letter is present and if the corresponding flag name is also mentioned
#         if expected_letter in gemini_response_lower:
#             # Find the flag name corresponding to the expected letter
#             correct_flag_name = next((flag for flag, letter in options_dict.items() if letter.lower() == expected_letter), None)
#             if correct_flag_name and correct_flag_name.lower().replace(" ", "") in gemini_response_lower:
#                 return True
#         return False
#     elif isinstance(correct_answer_info, list): # Q2 type
#         expected_letters = sorted([letter.lower() for letter in correct_answer_info])
#         # Try to extract letters from Gemini's response
#         found_letters = []
#         for letter in "abcdefghijklmnopqrstuvwxyz":
#             if letter + "." in gemini_response_lower or letter + "," in gemini_response_lower: # Account for variations
#                 found_letters.append(letter)
#         found_letters = sorted(found_letters)

#         # Check if the set of expected letters matches the set of found letters
#         if set(expected_letters) == set(found_letters):
#             # Additionally, check if the corresponding flag names are mentioned
#             correct_flag_names = [flag for flag, letter in options_dict.items() if letter.lower() in expected_letters]
#             all_flags_mentioned = True
#             for flag_name in correct_flag_names:
#                 if flag_name.lower().replace(" ", "") not in gemini_response_lower:
#                     all_flags_mentioned = False
#                     break
#             return all_flags_mentioned
#         return False
#     return False

# --- Main VQA Execution Loop ---
def perform_vqa_task(num_questions: int = 5, question_type: str = 'q1', geography_param: str = 'subregion'):
    """
    Executes the VQA task for a specified number of questions.
    """
    results = []

    for i in range(num_questions):
        image = get_image_from_hf_dataset(image_dataset, i)
        if image is None:
            print(f"Skipping question {i+1}: Could not load image.")
            continue

        mcq_options_dict = {}
        correct_answer_info = None

        try:
            if question_type == 'q1':
                mcq_options_dict, answer_key = generate_q1_options(i, flag_dataset, geography_param)
                if mcq_options_dict and answer_key:
                    correct_answer_info = answer_key[i]
                else:
                    print(f"Skipping question {i+1}: Failed to generate Q1 options.")
                    continue

            elif question_type == 'q2':
                mcq_options_dict, answer_key = generate_q2_options(i, flag_dataset, geography_param)
                if mcq_options_dict and answer_key:
                    correct_answer_info = answer_key[i]
                else:
                    print(f"Skipping question {i+1}: Failed to generate Q2 options.")
                    continue
            else:
                print(f"Skipping question {i+1}: Invalid question type '{question_type}'.")
                continue


        except Exception as e:
            print(f"Skipping question {i+1}: Error during MCQ generation: {e}")
            continue

        if not mcq_options_dict or not correct_answer_info:
            print(f"Skipping question {i+1}: No valid MCQ options or answer generated.")
            continue

        prompt = create_vqa_prompt(image, question_type, mcq_options_dict, correct_answer_info)

        #print(f"Question Prompt (first 200 chars):\n{prompt[:200]}...")
        # print(f"DEBUG: Correct Answer Info (letter(s)): {correct_answer_info}")
        # print(f"DEBUG: Options Dict: {mcq_options_dict}")

        gemini_answer = query_gemini_vqa(image, prompt)
        print(f"Gemini's Response: {gemini_answer}")

        is_correct = evaluate_gemini_answer(gemini_answer, mcq_options_dict, correct_answer_info)
        print(f"Expected Answer Letter(s): {correct_answer_info}")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        results.append(is_correct)

    correct_count = results.count(True)
    total_questions = len(results)
    if total_questions > 0:
        accuracy = (correct_count / total_questions) * 100
        print(f"\n--- VQA Task Summary ---")
        print(f"Total Questions Attempted: {total_questions}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo questions were successfully processed.")

if __name__ == "__main__":
    # Example usage for Q1
    perform_vqa_task(num_questions=5, question_type='q2', geography_param='subregion')

    # Example usage for Q2
    # perform_vqa_task(num_questions=5, question_type='q2', geography_param='continent')

