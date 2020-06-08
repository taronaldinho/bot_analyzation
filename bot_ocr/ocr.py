import numpy as np
import pandas as pd
import sys
import cv2
from PIL import Image
import pyocr
from pathlib import Path
from logging import (DEBUG, INFO, WARNING, ERROR, CRITICAL, Formatter,
                     StreamHandler, getLogger, handlers, log)


img_dir = r".\iCloud Photos"
output_dir =r""

logger_level         = DEBUG
stream_handler_level = DEBUG


handler_format = Formatter(
    "%(asctime)s %(levelname)-8s \
     [%(module)s.%(funcName)s %(lineno)d] %(message)s"
    )

# loggerの基本設定を行う。
logger = getLogger(__name__)
logger.setLevel(logger_level)
logger.propagate = False


# 標準出力へのログ出力設定を行う。
stream_handler = StreamHandler()
stream_handler.setLevel(stream_handler_level)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)



tools = pyocr.get_available_tools()
if len(tools) == 0:
    logger.critical("No OCR tool found. Script will be terminated.")
    sys.exit(1)

tool = tools[0]
logger.info("Will use tool '%s'" % (tool.get_name()))

langs = tool.get_available_languages()
logger.debug(langs)
lang = langs[langs.index("jpn")]
logger.info("Will use lang '%s'" % (lang))

builder = pyocr.builders.DigitLineBoxBuilder(tesseract_layout=4)


p_original = np.float32([[75, 181], [890, 158], [66, 618], [884, 677]])
p_trans = np.float32([[100, 150], [900, 150], [100,670], [900, 670]])
M = cv2.getPerspectiveTransform(p_original, p_trans)

areas = {"RANK": [80, 140], "BATTLES": [550, 620],
         "WIN": [630, 700], "LOSE": [710, 760],
         "DRAW":[780, 830], "SCORE":[840, 900]}



target_dir = Path(img_dir)
df = pd.DataFrame()
for img_path in target_dir.iterdir():
    if img_path.suffix.lower() != ".png":
        continue

    logger.debug(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    logger.debug(f"raw image shape| w: {w}, h: {h}")


    i_trans = cv2.warpPerspective(img, M, (900, 680), flags=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC
    h, w = i_trans.shape
    logger.debug(f"i_trans shape| w: {w}, h: {h}")

    i_masked = cv2.rectangle(i_trans, (0, 0), (900, 150), (0), thickness=-1)
    i_masked = cv2.rectangle(i_trans, (0, 0), (95, 680), (0), thickness=-1)
    i_masked = cv2.rectangle(i_trans, (150, 0), (530, 680), (0), thickness=-1)

    _, i_bin = cv2.threshold(i_masked, 130, 255, cv2.THRESH_BINARY_INV)


    # cv2.imwrite(str(img_path) + "t" + ".png",  i_bin)

    ret = dict()
    for k, v in areas.items():
        i_t = i_bin[:, v[0] : v[1]]
        h, w = i_t.shape
        logger.debug(f"i_trans shape| w: {w}, h: {h}")

        txt = tool.image_to_string(
            Image.fromarray(i_t, "L"),
            lang=lang,
            builder=builder
        )

        if len(txt) != 0:
            pass
        else:
            logger.warning(f"{img_path.name}, {k}: no text detected.")

        buf = ""
        val = []
        for t in txt:
            
            buf += t.content+"\n"

        buf = buf[:-1]
        val = buf.split("\n")
        
        if len(val) < 10:
            val.extend(["" for i in range(10 - len(val))])
        
        ret[k] = val

    raw_df = pd.DataFrame(ret)
    raw_df["file"] = str(img_path)    
    df = pd.concat([df, raw_df])

df["v_0"] = False
df.loc[df["BATTLES"] == df["WIN"] + df["LOSE"] + df["DRAW"], "v_0"] = True
# df["RANK"] = df["RANK"].apply(np.abs)
# df.astype(int)


with open("raw_result.csv", mode="w", encoding="cp932", newline="", errors="ignore") as f:
    df.to_csv(f)