# Commerce Pulse dashboard

Interactive Streamlit dashboard for the e-commerce analysis in `class_task.ipynb`.

## Run it

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app looks for `ecommerce_sales_data.csv` in this folder or `data/`. You can also upload the CSV in the left sidebar. The notebook CSV was not present when the dashboard was created, so the app uses a clearly labeled illustrative dataset until a real CSV is supplied.

The dashboard recognizes `Revenue` (or `Sales` / `Amount`), `Profit`, `Units_Sold` (or `Quantity`), `Ad_Spend` (or `Marketing Spend`), `Category`, `Region`, and either a date column or `Year` plus `Month`. It includes overview KPIs and trends, product and regional rankings, marketing efficiency, data-quality metrics, row search, and filtered CSV download.
