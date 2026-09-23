from langchain_community.document_loaders import PyPDFLoader

loader =  PyPDFLoader('7.Langchain_documentLoader/dl-curriculum.pdf')

docs = loader.load()

# print(docs[0].page_content)

print(len(docs))