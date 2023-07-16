# %%
import os
import openai
from typing import List

test_case = {
    "input_text": "San Francisco and the surrounding San Francisco Bay Area are a global center of economic activity and the arts and sciences,[34][35] spurred by leading universities,[36] high-tech, healthcare, finance, insurance, real estate, and professional services sectors.[37] As of 2020, the metropolitan area, with 6.7 million residents, ranked 5th by GDP ($874 billion) and 2nd by GDP per capita ($131,082) across the OECD countries, ahead of global cities like Paris, London, and Singapore.[38][39][40] San Francisco anchors the 13th most populous metropolitan statistical area in the United States with 4.6 million residents, and the fourth-largest by aggregate income and economic output, with a GDP of $669 billion in 2021.[41] The wider San Jose–San Francisco–Oakland Combined Statistical Area is the fifth most populous, with 9.5 million residents, and the third-largest by economic output, with a GDP of $1.25 trillion in 2021. In the same year, San Francisco proper had a GDP of $236.4 billion, and a GDP per capita of $289,990.[41] San Francisco was ranked seventh in the world and third in the United States on the Global Financial Centres Index as of March 2022.",
    "questions": [
        "How many residents lived in San Francisco in 2020?",
        "Which city had a greater GDP in 2020: Paris or San Francisco?",
        "What's Nick's favorite color?",
    ],
}


class TextChatBot:
    """
    Summary: Asks OpenAI's gpt-3.5-turbo to answer a list of questions using only a provided input text.
    Returns "out of scope" if ChatGPT believes that the given question is not answerable using only the input text.

    :param str input_text: A string of input text that should be used to answer the user's questions.
    :param list questions: A list of questions we want GPT to answer
    :return: A list of answers, one for each of the user's questions
    """

    def __new__(cls, input_text: str, questions: list):
        """Allow this class to be called like a function, or instantiated like a class."""
        instance = super().__new__(cls)
        if not input_text and not questions:
            return instance
        else:
            return instance(input_text, questions)

    def __call__(self, input_text: str, questions: list) -> list:
        """Our main function: return a list of answers to a list of questions about some given input text"""

        assert isinstance(
            input_text, str
        ), f"Input text is of the wrong type; expected str but got {type(input_text).__name__}"

        assert isinstance(
            questions, list
        ), f"Question list is of the wrong type; expected list but got {type(questions).__name__}"

        self._set_openai_api_key()

        output = []
        for question in questions:
            response = self._query_gpt(input_text, question)
            output.append(response)

        return output

    def _set_openai_api_key(self):
        """Set the OpenAI API key for use in the session."""
        API_KEY = os.getenv("API_KEY")
        openai.api_key = API_KEY

    def _get_prompt_text(self, input_text: str, question: str):
        """Transform the user's input_text and question into a formatted prompt that explains the rules for GPT's response"""

        return f"""
        Pretend that you can only answer questions about the following text: {input_text}

        Answer the following question using only the context contained in the previous text: {question}

        If the question above is unrelated to the question in the text, then respond with "out of scope". Otherwise, answer the question in one complete sentence.
        """

    def _query_gpt(self, input_text: str, question: str) -> str:
        """Queries gpt-3.5-turbo to answer a given question using some input text"""
        prompt_template = self._get_prompt_text(
            input_text=input_text, question=question
        )

        assert (
            len(prompt_template.split(" ")) < 2048
        ), f"Prompt is too long to process via the GPT API. Try shortening your input_text and/or question"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_template}],
        )

        answer = response.choices[0].message.content

        return answer


TextChatBot(test_case["input_text"], test_case["questions"])

# %%
