from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return the first 10 numbers of the fibonacci sequence", type=str)
parser.add_argument("--language", default="python", type=str)
args = parser.parse_args()

llm = ChatOpenAI()

code_prompt = PromptTemplate(
  template = "Write a very short {language} function that will {task}",
  input_variables=["language", "task"],
);

test_prompt = PromptTemplate(
  template = "Write a test for the following {language} code:\n{code}",
  input_variables=["language", "code"],
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  output_key="code",
);

test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key="test",
);

chain  = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=["language", "task"],
  output_variables=["code", "test"],
)

result = chain({
  "language": args.language,
  "task": args.task
})

print(">>> Generated Code:")
print(result["code"])

print(">>> Generated Test:")
print(result["test"])