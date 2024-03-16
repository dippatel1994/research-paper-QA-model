import gradio as gr
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering


class ResearchPaperQAModel:
    """Class to load the model and answer questions based on abstract and text of reserach paper.
    """
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer_question(self, question, abstract, paper_text):
        # Tokenize input question and context
        if not paper_text:
            context = abstract
        else:
            context = paper_text
            
        inputs = self.tokenizer(question, context, return_tensors="tf")

        # Get the start and end logits for the answer
        outputs = self.model(**inputs)
        start_logits, end_logits = outputs.start_logits[0].numpy(), outputs.end_logits[0].numpy()

        # Find the tokens with the highest probability for start and end positions
        start_index = tf.argmax(start_logits, axis=-1).numpy()
        end_index = tf.argmax(end_logits, axis=-1).numpy()

        # Convert token indices to actual tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy().squeeze())
        answer_tokens = tokens[start_index : end_index + 1]

        # Convert answer tokens back to a string
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        return answer


model = "bert-large-uncased-whole-word-masking-finetuned-squad" # Model name
paper_model = ResearchPaperQAModel(model) #Create an instance of the model

# Create a Gradio interface
iface = gr.Interface(
    fn=paper_model.answer_question,
    inputs=["text", "text", "text"],
    outputs="text",
    live=True,
    title="Ask question to research paper",
    description="Enter title of research paper, abstract, research paper content(optional) and list of questions to get answers."
)

# Launch the Gradio interface
iface.launch(share=True)
