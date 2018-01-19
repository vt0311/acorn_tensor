from PIL import Image
import numpy as np
import os, re
 
# 파일 경로 지정하기
search_dir = "./image"
cache_dir = "./image/cache_avhash"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
 
# 이미지 데이터를 Average Hash로 변환하기 --- (※1)
def average_hash(fname, size = 16):
    # fname : 비교 대상 파일의 전체 경로 + 파일 이름
    # print('fname : ', fname)
    # print('폴더 문자열 길이 :', len(search_dir))
    fname2 = fname[len(search_dir):] # 문자열 슬라이싱
    # print('파일 이름 : ', fname2) # 파일 이름 예시 : /dragonfly/image_0029.jpg
 
    # cache_file(이미지 캐시하기) : 해당 그림에 대한 픽셀에 대한 정보를 담고 있는 파일
    cache_file = cache_dir + "/" + fname2.replace('/', '_') + ".csv"
    # print('cache_file : ', cache_file)
    if not os.path.exists(cache_file): # 해시 생성하기
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.ANTIALIAS)
        pixels = np.array(img.getdata()).reshape((size, size))
        avg = pixels.mean()
        px = 1 * (pixels > avg)
 
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else: # 캐시돼 있다면 읽지 않기
        px = np.loadtxt(cache_file, delimiter=",")
    return px
 
# 해밍 거리 구하기 --- (※2)
def hamming_dist(a, b):
    aa = a.reshape(1, -1) # 1차원 배열로 변환하기
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist
 
# 모든 폴더에 처리 적용하기 --- (※3)
def enum_all_files(path):
    # os.walk : 하위 디렉토리를 쉽게 검색할 수 있다.
    for root, dirs, files in os.walk(path):
        
        for f in files:  # 모든 파일에 대하여 ...
            # os.path.join : 디렉토리 이름과 파일 이름을 연결하여 전체 경로를 만들어 준다.
            fname = os.path.join(root, f)
            
            # jpg파일이나 png파일에 대하여...
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname # yield : 제너레이터 함수로 만들기(for 반복문 사용이 가능해짐)
 
# 이미지 찾기 --- (※4)
def find_image(fname, rate):
    
    src = average_hash(fname) # 원본 그림
    # print( 'src : ', src )
    
    for fname in enum_all_files(search_dir):
        fname = fname.replace('\\', '/')
        
        dst = average_hash(fname)  # 비교 대상 그림
        # print('dst : ', dst)
        
        # 해밍 거리를 구하고...
        diff_r = hamming_dist(src, dst) / 256
        # print("[check] ",fname)
        
         # 기준으로 잡은 거리보다 작으면... (rate = 0.25 )
         # why : 해밍 거리는 불일치하는 것을 기준으로 잡으므로...
        if diff_r < rate: 
            yield (diff_r, fname)
 
# 찾기 --- (※5) # "./image/101_ObjectCategories"
srcfile = search_dir + "/image_0016.jpg"
html = ""
 
# 0.25 : 75% 이상의 픽셀이 동일한 것 모두 찾아라.
sim = list(find_image(srcfile, 0.25))
# 유사도가 높은 항목이 앞으로 오도록 정렬...
sim = sorted(sim, key=lambda x:x[0])
 
for r, f in sim:
    # r : 유사도, f : 해당 파일 이름
    print(r, ">", f)
    newr = str((1 - r ) * 100) + '%'
 
    # os.path.basename : 파일명을 분리해준다.
    s = '<div style="float:left;"><h3>[ 유사율 : ' + newr + '-' + \
        os.path.basename(f) + ']</h3>'+ \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>'+ \
        '</a></p></div>'
    html += s # s하나를 그림 한개라고 보면 된다.
 
# HTML로 출력하기
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)
 
# avhash-search-output.html :
with open("./avhash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)
 
print("ok")