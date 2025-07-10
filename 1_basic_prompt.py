from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(model="o3-mini-2025-01-31")


# Defining the system prompt (how the AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."
)

# the user prompt is provided by the user, in this case however the only dynamic
# input is the article
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for a article.
The article is here for you to examine {article}

The name should be based of the context of the article.
Be creative, but make sure the names are clear, catchy,
and relevant to the theme of the article.

Only output the article name, no other explanation or
text can be provided.""",
    input_variables=["article"],
)
# user_prompt.format(article="TEST STRING")

prompt_template = ChatPromptTemplate([system_prompt, user_prompt])

pipeline = prompt_template | llm
article = "This is the age of AI"
# article_title = pipeline.invoke({"article": article}).content
second_user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a description forthe article. The article is here for you to examine:
---
{article}
---
Here is the article title '{article_title}'.
Output the SEO friendly article description. Do not output anything other than the description.""",
    input_variables=["article", "article_title"],
)
second_prompt = ChatPromptTemplate.from_messages([system_prompt, second_user_prompt])

chain_two = (
    {"article": lambda x: x["article"], "article_title": lambda x: x["article_title"]}
    | second_prompt
    | llm
    | {"summary": lambda x: x.content}
)

# article_description = chain_two.invoke(
#     {"article": article, "article_title": article_title}
# )
# print(f"Article Title:{article_title}")
# print(f"Article Description: {article_description['summary']}")


class Schema(BaseModel):
    article: str = Field("The original Article provided by the user")
    article_title: str = Field("Title of the article")


schema = Schema(article="Adad", article_title="adsada")
print(schema.model_dump())

llm_with_structure = llm.with_structured_output(Schema)
# # RunnablesPassthrough ->
# # langgraph -> State ->
combined_chain = (
    {"article": lambda x: x["article"]}
    | prompt_template
    | llm_with_structure
    | {
        "article": lambda x: x.model_dump()["article"],
        "article_title": lambda x: x.dict()["article_title"],
    }
    | second_prompt
    | llm
    | {"summary": lambda x: x.content}
)


# structure_output = llm_with_structure.invoke(
#     "YOu are given a article: <Article>This is era of AI where people use AI to get done with their day to day tasks </Article> Your task is to genrate a SEO friendly title for the article"
# )

print(combined_chain.invoke({"article": article}))
