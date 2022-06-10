import string
import pandas as pd


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
from haystack.nodes import BM25Retriever, EmbeddingRetriever

from haystack import Pipeline, JoinDocuments


content_dicts = load_data()
print(len(content_dicts))

bm_document_store = ElasticsearchDocumentStore(similarity="cosine",
											   embedding_dim=384, index="bm_documents")
bm_document_store.delete_documents()
bm_document_store.write_documents(content_dicts)

bm_retriever = BM25Retriever(document_store=bm_document_store)

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

print("begin update es_document_store embeddings")
es_document_store.update_embeddings(e_retriever)
print("end update es_document_store embeddings")

combined_p = Pipeline()
combined_p.add_node(component=bm_retriever, name="BMRetriever",
					inputs=["Query"])

combined_p.add_node(component=e_retriever, name="ESRetriever",
					inputs=["Query"])

combined_p.add_node(component = JoinDocuments(join_mode="merge", weights = [0.5, 0.5]),
					name="JoinResults_content",
					inputs=["BMRetriever", "ESRetriever"])


if __name__ == '__main__':

	query = "does covid causes death"
	result = combined_p.run(
		query=clean_text(query),
		params={
			"BMRetriever": {"top_k": 3},
			"ESRetriever": {"top_k": 3}
		})

	for r in result['documents']:
		print(r.meta['question'])
		print(r.meta['answers'])
		print(r.score)
		print()
