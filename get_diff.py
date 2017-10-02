import os, re, requests, sys

if len(sys.argv) < 2:
    print('Usage ', sys.argv[0], ' <reg_url>')
    sys.exit(1)

url = sys.argv[1]

base_url = re.sub('index.html$', '', url)
json_url = re.sub('index.html$', 'out.json', url)
json = requests.get(json_url).json()

for item in json['failedItems']:
    img1 = base_url + 'actual' + item
    img2 = base_url + 'expected' + item
    img3 = base_url + 'diff' + item
    paths = item.split('/')[1:-1]
    os.makedirs('img/actual/' + os.path.join(*paths), exist_ok=True)
    os.makedirs('img/expected/' + os.path.join(*paths), exist_ok=True)
    os.makedirs('img/diff/' + os.path.join(*paths), exist_ok=True)
    print('Download ', item)
    with open('img/actual' + item, 'wb') as fout:
        fout.write(requests.get(img1).content)
    with open('img/expected' + item, 'wb') as fout:
        fout.write(requests.get(img2).content)
    with open('img/diff' + item, 'wb') as fout:
        fout.write(requests.get(img3).content)
