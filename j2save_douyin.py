import requests
from bs4 import BeautifulSoup

def save_video_douyin(url):
    cookies = {
        'fpestid': 'nn6HSJ89WCMhYTN1sGgMEAQj5M4SKiRxEVJvsbpSA1jj8m-JvsUyxM4fi6sYT9havDRY1Q',
        's_id': 'f8U2qlmonIWUEnvfRaIqrG2urjLMl8cq0BnfQQuo',
        'XSRF-TOKEN': 'eyJpdiI6IkFUZjVDdFNxS1daSWVmVnVSVEQ2b0E9PSIsInZhbHVlIjoiQWN2OVBtQ0tIY2NqdGpwVDVcL3o2VzRNMEZcL25ISVNoeFEraVpld2Q4S1U4bjFkQlA2dDhLSTYzRDlpNHFkYXJzIiwibWFjIjoiZTFiODQ2YjQ3ZTc4MjkyYWJhNDM3ZTVmNDI1MDM4MWY4ZTk5YTdhODgwZGVmZWQ1YzdmOTU0MTc1YmE5MDVmYiJ9',
    }

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'vi,en-US;q=0.9,en;q=0.8,fr-FR;q=0.7,fr;q=0.6',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        # 'cookie': 'fpestid=nn6HSJ89WCMhYTN1sGgMEAQj5M4SKiRxEVJvsbpSA1jj8m-JvsUyxM4fi6sYT9havDRY1Q; s_id=f8U2qlmonIWUEnvfRaIqrG2urjLMl8cq0BnfQQuo; XSRF-TOKEN=eyJpdiI6IkFUZjVDdFNxS1daSWVmVnVSVEQ2b0E9PSIsInZhbHVlIjoiQWN2OVBtQ0tIY2NqdGpwVDVcL3o2VzRNMEZcL25ISVNoeFEraVpld2Q4S1U4bjFkQlA2dDhLSTYzRDlpNHFkYXJzIiwibWFjIjoiZTFiODQ2YjQ3ZTc4MjkyYWJhNDM3ZTVmNDI1MDM4MWY4ZTk5YTdhODgwZGVmZWQ1YzdmOTU0MTc1YmE5MDVmYiJ9',
    }

    params = {
        'search': url,
    }

    response = requests.get('https://downloader.twdown.online/search', params=params, cookies=cookies, headers=headers)
    soup=BeautifulSoup(response.text, 'html.parser');link=''
    while True:
        for i in  (soup.find_all('a')):
            if  (i.get('href')).find('https://downloader.twdown.online?ref=&title=+#url=') != -1:
                link=(i.get('href'))
                break       
        if link !='':
            break                                  
    headers = {
        'accept': '*/*',
        'accept-language': 'vi,en-US;q=0.9,en;q=0.8,fr-FR;q=0.7,fr;q=0.6',
        'priority': 'u=1, i',
        'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
        'cookie': 'fpestid=nn6HSJ89WCMhYTN1sGgMEAQj5M4SKiRxEVJvsbpSA1jj8m-JvsUyxM4fi6sYT9havDRY1Q; s_id=r8PPssTZzRcI5SFtLngc80z8qbGh6hTjB0WwGRUh; XSRF-TOKEN=eyJpdiI6IkpPakYwbjN0WE5CVGFncVJTczREelE9PSIsInZhbHVlIjoiRGgyNk5uRHBLNjZTcHcwMTZcL1BFTlFxMm92NVJUd1MzUExOcyt0bldwNkQzM1k0UnNKQ0dSXC9pN3gxbHVicDg5IiwibWFjIjoiMzllNjhkY2VmMjYzNTk5M2ZlNWRkMDZhZjk1MzdjZTEzN2NiNGI1NDExNWRmZGM1ZmVmNGYxMjhlNGEyOGQ2NiJ9',

    }

    params = {
        'url': link.replace('https://downloader.twdown.online?ref=&title=+#url=',''),
    }

    response = requests.get('https://downloader.twdown.online/load_url', params=params, headers=headers)
    link_video=(response.text)
    return link_video    



