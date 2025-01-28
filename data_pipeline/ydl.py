import os 
from yt_dlp import YoutubeDL

############################################################################
challenge_name = "Maratanghuru"
output_path = os.path.join("./yt_videos", challenge_name)
urls = {
    'right_video': 'https://www.youtube.com/shorts/97BVtGb21lU',
    'wrong_videos': [
        'https://www.youtube.com/shorts/COwRJMCCWL0',
        'https://www.youtube.com/shorts/VGB-mC9xqkE',
        'https://www.youtube.com/shorts/vA5Vd6y-MOk',
        'https://www.youtube.com/shorts/kiMz8T38_8g',
        'https://www.youtube.com/shorts/ExI5USR7xp0'
    ]
}
############################################################################


# 다운로드 함수
def download_video(url, output_path, file_name):
    """
    유튜브 영상을 다운로드하여 지정된 경로에 저장합니다.

    :param url: 유튜브 영상 URL
    :param output_path: 저장할 디렉토리 경로
    """
    # 지정된 경로가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # yt-dlp 옵션 설정
    ydl_opts = {
        'outtmpl': os.path.join(output_path, file_name + '.%(ext)s'),  # 파일명 템플릿
        'format': 'bestvideo',  # 최고 화질 선택
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])  # URL 리스트 전달

def download_challenge(challenge_name, output_path, urls):
    '''
    challenge_name : 챌린지의 이름. 폴더별로 영상 구분하기 위해 설정
    output_path : 비디오를 저장할 최상위 폴더. 비디오는 output_path/challenge_name/... 에 저장됨
    urls : key값으로 right_video, wrong_videos 2가지를 가짐. right_video는 정답 video의 url/ wrong_videos는 비교대상들의 url list
    '''
    output_path = os.path.join(output_path, challenge_name)

    right_video_url = urls['right_video']
    download_video(right_video_url, output_path, f'right_video')

    wrong_video_urls = urls['wrong_videos']
    for idx, url in enumerate(wrong_video_urls):
        download_video(url, output_path, f'wrong_video_{idx}')


def main():
    right_video_url = urls['right_video']
    download_video(right_video_url, output_path, f'right_video')

    wrong_video_urls = urls['wrong_videos']
    for idx, url in enumerate(wrong_video_urls):
        download_video(url, output_path, f'wrong_video_{idx}')


if __name__ == "__main__":
    main()