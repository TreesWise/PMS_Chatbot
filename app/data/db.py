# from sqlalchemy import create_engine
# import urllib

# def get_db_engine():
    
#     params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=10.201.1.86,50001;"
#     "DATABASE=Resume_Parser;"
#     "Trusted_Connection=Yes;"
# )
    
    
#     return create_engine(
#        f"mssql+pyodbc:///?odbc_connect={params}"
#     )



from sqlalchemy import create_engine

def get_db_engine():
    return create_engine(
        "mssql+pyodbc://@10.201.1.86,50001/Resume_Parser"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    )