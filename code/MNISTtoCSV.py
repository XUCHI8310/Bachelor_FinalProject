def process(img, label, out, n):
    """

    @param img:
    @param label:
    @param out:
    @param n:
    @return: train & test csv file
    """

    f = open(img, "rb")  # Read only, binary
    o = open(out, "w")  # Write only
    l = open(label, "rb")

    f.read(16)
    l.read(8)
    images = []  # Init

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()
