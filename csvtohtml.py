import pandas as pd

# Read the CSV file
df = pd.read_csv('leave_requests.csv')

# Convert DataFrame to HTML
html_table = df.to_html(index=False)

# Create an HTML file and write the table into it
with open('templates/leave_request.html', 'w') as file:
    file.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Leave Requests</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>Leave Requests</h1>
        ''' + html_table + '''
    </body>
    </html>
    ''')
