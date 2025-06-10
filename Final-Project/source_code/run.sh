python main.py --query-csv-path "cleaned_final_project_query.csv" --output-csv-path "output_rag.csv" --output-info-csv-path "output_info_rag.csv" --use-rag 1 --print-response 0
python main.py --query-csv-path "cleaned_final_project_query.csv" --output-csv-path "output_without_rag.csv" --output-info-csv-path "output_info_without_rag.csv" --use-rag 0 --print-response 0

python vote.py
