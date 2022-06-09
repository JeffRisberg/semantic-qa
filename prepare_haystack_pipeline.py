import string

import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever


def clean_text(text):
	return ''.join(x for x in text.lower() if x in string.printable).replace(
		'.', '')


def load_data():
	content_dicts = []

	df = pd.read_csv("data/COVID-QA.csv")
	df.dropna(inplace=True, subset=["Category", "Question ID", "Answers", "Question"])

	print("\nLoading QA's...")
	for i, row in df.iterrows():
		content_dict = {
			'content': clean_text(row['Question']),
			'meta': {'q_id': row['Question ID'],
					 "question": row['Question'],
					 "answers": row['Answers'],
					 "category": row['Category'],
					 "row": i
					 }
		}
		content_dicts.append(content_dict)
	return content_dicts


from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Pipeline

if __name__ == '__main__':
	print("Model loaded successfully...")
	content_dicts = load_data()
	print(len(content_dicts))

	tf_document_store = InMemoryDocumentStore()
	tf_document_store.delete_documents()
	tf_document_store.write_documents(content_dicts)

	tf_retriever = TfidfRetriever(document_store=tf_document_store,
								  auto_fit=True)

	es_document_store = ElasticsearchDocumentStore(similarity="cosine",
												   embedding_dim=384,
												   index="document")
	es_document_store.delete_documents()
	es_document_store.write_documents(content_dicts)

	model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
	e_retriever = EmbeddingRetriever(
		document_store=es_document_store,
		embedding_model=model_name
	)

	es_document_store.update_embeddings(e_retriever)

	combined_p = Pipeline()
	combined_p.add_node(component=tf_retriever, name="TFRetriever",
						inputs=["Query"])
	combined_p.add_node(component=e_retriever, name="ERetriever",
						inputs=["Query"])

	query = "does covid causes death"
	result = combined_p.run(
		query=clean_text(query),
		params={
			"TFRetriever": {"top_k": 3},
			"ERetriever": {"top_k": 3}
		})

	for r in result['documents']:
		print(r.meta['question'])
		print(r.meta['answers'])
		print(r.score)
		print()
