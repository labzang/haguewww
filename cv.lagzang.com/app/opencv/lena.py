import cv2
import os


class LenaModel:
    def __init__(self):
        # 현재 파일의 디렉토리를 기준으로 data/opencv 경로 설정
        base_dir = os.path.dirname(os.path.dirname(__file__))
        opencv_data_dir = os.path.join(base_dir, "data", "opencv")
        self.fname = os.path.join(opencv_data_dir, "lena.jpg")

    def execute(self):
        original = cv2.imread(self.fname, cv2.IMREAD_COLOR)
        gray = cv2.imread(self.fname, cv2.IMREAD_GRAYSCALE)
        unchanged = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)
        """
        이미지 읽기에는 위 3가지 속성이 존재함.
        대신에 1, 0, -1 을 사용해도 됨.
        """
        cv2.imshow("Original", original)
        cv2.imshow("Gray", gray)
        cv2.imshow("Unchanged", unchanged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # 윈도우종료


if __name__ == "__main__":
    lena = LenaModel()
    lena.execute()
