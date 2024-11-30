import os
import shutil
from glob import glob


def split_dataset(input_dir, output_dir):
    """
    input_dir: str: 원본 이미지들이 저장된 디렉토리 경로
    output_dir: str: train과 val 디렉토리를 생성할 경로
    """
    # 데이터셋 경로 설정
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # 디렉토리 생성
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 이미지 파일 리스트 가져오기
    image_files = glob(os.path.join(input_dir, '*.jpg')) + \
                  glob(os.path.join(input_dir, '*.png')) + \
                  glob(os.path.join(input_dir, '*.jpeg'))

    # 데이터셋 분할
    train_files = []
    val_files = []

    for i, file in enumerate(image_files):
        if i % 5 == 0:
            val_files.append(file)
        else:
            train_files.append(file)

    # train 이미지 복사
    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))

    # validation 이미지 복사
    for file in val_files:
        shutil.copy(file, os.path.join(val_dir, os.path.basename(file)))

    print(f'Total images: {len(image_files)}')
    print(f'Train images: {len(train_files)}')
    print(f'Validation images: {len(val_files)}')


# 사용 예제
input_dir_A = 'D:\cyclegan\pytorch-CycleGAN-and-pix2pix/results/new_resize/test_latest\images\A'  # 입력 데이터셋 디렉토리 A
input_dir_B = 'D:\cyclegan\pytorch-CycleGAN-and-pix2pix/results/new_resize/test_latest\images\B'  # 입력 데이터셋 디렉토리 B
output_dir_A = 'D:\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets/newconvert/A'  # 출력 데이터셋 디렉토리 A
output_dir_B = 'D:\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets/newconvert/B'  # 출력 데이터셋 디렉토리 B

# 디렉토리 A와 B 각각에 대해 데이터셋 분할 수행
split_dataset(input_dir_A, output_dir_A)
split_dataset(input_dir_B, output_dir_B)
