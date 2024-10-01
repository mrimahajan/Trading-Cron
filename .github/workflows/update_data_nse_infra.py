name: Update NSE Data Infra

on:
  schedule:
    - cron: '*/1 * * * *'  # This is the cron expression to run every 5 minutes

jobs:
  my_job:
    runs-on: ubuntu-latest  # You can change this based on your environment
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          pip install django
          pip install yfinance
          pip install psycopg2

      - name: Run script or command
        run: python update_data_nse_infra.py
