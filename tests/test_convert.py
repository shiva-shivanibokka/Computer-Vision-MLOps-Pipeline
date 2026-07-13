from PIL import Image

from cvmlops.data.convert_hripcb import convert

VOC_XML = """<annotation>
  <filename>{name}.jpg</filename>
  <size><width>200</width><height>100</height></size>
  <object><name>short</name>
    <bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox>
  </object>
  <object><name>missing_hole</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>10</ymax></bndbox>
  </object>
  <object><name>not_a_real_class</name>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox>
  </object>
</annotation>"""


def test_voc_to_yolo_conversion(tmp_path):
    src = tmp_path / "PCB_DATASET"
    (src / "images").mkdir(parents=True)
    (src / "ann").mkdir(parents=True)
    Image.new("RGB", (200, 100), (10, 80, 40)).save(src / "images" / "b01.jpg")
    (src / "ann" / "b01.xml").write_text(VOC_XML.format(name="b01"))

    out = convert(src, tmp_path / "raw")
    lines = (out / "labels" / "b01.txt").read_text().splitlines()

    # unknown class dropped -> 2 boxes, not 3
    assert len(lines) == 2

    # short: center (100,40)/(200,100) = 0.5,0.4 ; size 100x40 -> 0.5,0.4
    cls, cx, cy, bw, bh = lines[0].split()
    assert cls == "3"  # index of "short" in params classes
    assert (float(cx), float(cy), float(bw), float(bh)) == (0.5, 0.4, 0.5, 0.4)

    # missing_hole is class index 0
    assert lines[1].split()[0] == "0"
    assert (out / "images" / "b01.jpg").exists()
