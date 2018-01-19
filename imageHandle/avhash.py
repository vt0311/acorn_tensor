from PIL import Image
import numpy as np

# 이미지 데이터를 Average Hash로 변환하기 --- (※1)
def average_hash(fname, size = 16):
    img = Image.open(fname) # 이미지 데이터 열기---(※2)
    img = img.convert('L') # 그레이스케일로 변환하기 --- (※3)
    img = img.resize((size, size), Image.ANTIALIAS) # 리사이즈하기 --- (※4)

    pixel_data = img.getdata() # 픽셀 데이터 가져오기 --- (※5)
    print(type(pixel_data)) # <class 'ImagingCore'>

    pixels = np.array(pixel_data) # Numpy 배열로 변환하기 --- (※6)
    print(pixels.ndim, '/', pixels.shape)  # 1 / (256,)

    pixels = pixels.reshape((size, size)) # 2차원 배열로 변환하기 --- (※7)
    print(pixels.ndim, '/', pixels.shape) # 2 / (16, 16)
    avg = pixels.mean() # 각 픽셀(총 256개)의 산술 평균 구하기 --- (※8)
    print('평균 픽셀 : ', avg ) # 평균 픽셀 :  57.10546875
    print('\n각 픽셀의 값 출력')
    print(pixels)
    print('\n')
    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로 변환하기 --- (※9)
    return diff

# 이진 해시로 변환하기 --- (※10)
def np2hash(n):
    bhash = []
    # ahash.tolist() : 16개의 리스트 목록을 담고 있는 리스트 객체
    for nl in ahash.tolist():
        sl = [str(i) for i in nl]
        s2 = "".join(sl) # join() 메소드 : 특정 문자열에 다른 문자열을 추가한다.
        # print('s2 : ', s2)
        i = int(s2, 2) # 이진수를 십진 정수로 변환하기
        bhash.append("%04x" % i) # %d : 10진수, %x : 16진수
    return "".join(bhash)

# Average Hash 출력하기
ahash = average_hash('tower.jpg')
print(ahash, '\n')
print(np2hash(ahash), '\n')