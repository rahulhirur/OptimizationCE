import pandas as pd
from scholarly import scholarly, ProxyGenerator

def fetch_google_scholar_author_profile(author_name):
    # Initialize ProxyGenerator (optional, in case you need to use proxies)
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)
   
    # Search for the author by name
    search_query = scholarly.search_author(author_name)
    author = next(search_query, None)
   
    if not author:
        print(f"No author found for name: {author_name}")
        return None
   
    # Fill the author information
    author = scholarly.fill(author)

    print(author.get('email_domain'),"auth", author_name)

    # Extract the required information
    author_info = {
        'name':author_name,
    'email_domain': author.get('email_domain'),
    'affiliation': author.get('interests', []),
    'Cited by': author.get('citedby', 0)
    }


    return author_info


df = pd.read_excel("/Users/usi/Documents/Optimization/LAB02/authorData.xlsx")
authors_unique = df['Authors']

print(authors_unique)
author_profile = []
for nam in authors_unique:
    author_profile = fetch_google_scholar_author_profile(nam)

