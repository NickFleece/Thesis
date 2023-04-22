from ultralytics import YOLO

model = YOLO("yolov8x-pose.pt")

model.predict(R"D:\JHMDB\JHMDB_video\ReCompress_Videos\brush_hair\April_09_brush_hair_u_nm_np1_ba_goo_0.avi", save=True, save_txt=True)
