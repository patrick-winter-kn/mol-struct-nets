import argparse
import os
from os import path
import imageio
import tempfile
import shutil
import cairosvg
import math
from util import process_pool, multi_process_progressbar


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates an animated GIF from the SVGs contained in a directory')
    parser.add_argument('input', type=str, help='The input directory containing the SVGs')
    parser.add_argument('output', type=str, help='Path of the animated GIF')
    parser.add_argument('--size', type=int, default=1024, help='The size in pixels')
    parser.add_argument('--duration', type=float, default=0.2, help='The duration of each frame in seconds')
    parser.add_argument('--numbered', type=bool, default=False,
                        help='If True the file names are expected as 0.svgz, 1.svgz, ...')
    return parser.parse_args()


def convert_svgs_to_pngs(in_out_pairs, width, height, progress=None):
    for pair in in_out_pairs:
        cairosvg.svg2png(url=pair[0], write_to=pair[1], parent_width=width, parent_height=height)
        if progress is not None:
            progress.increment()


def chunk(number, number_chunks):
    chunks = []
    chunk_size = math.ceil(number / number_chunks)
    number_chunks = math.ceil(number / chunk_size)
    for i in range(number_chunks):
        start = chunk_size * i
        end = min(start + chunk_size, number)
        size = end - start
        chunks.append({'size': size, 'start': start, 'end': end})
    return chunks


args = get_arguments()
input_dir = path.abspath(args.input)
output_file = path.abspath(args.output)
temp_dir = tempfile.mkdtemp(prefix='svgs_to_gif')
if args.numbered:
    svgs = list()
    for i in range(len(os.listdir(input_dir))):
        svgs.append(str(i) + '.svgz')
else:
    svgs = os.listdir(input_dir)
    svgs = sorted(svgs)
in_out_pairs = list()
images = list()
for svg in svgs:
    output_file_path = temp_dir + os.sep + svg[svg.rfind('/') + 1:svg.rfind('.')] + '.png'
    svg_path = input_dir + os.sep + svg
    in_out_pairs.append((svg_path, output_file_path))
    images.append(output_file_path)
chunks = chunk(len(images), process_pool.default_number_processes)
print('Converting SVGs')
with process_pool.ProcessPool(len(chunks)) as pool:
    with multi_process_progressbar.MultiProcessProgressbar(len(svgs)) as progress:
        for chunk in chunks:
            pool.submit(convert_svgs_to_pngs, in_out_pairs[chunk['start']:chunk['end']], args.size, args.size,
                        progress.get_slave())
        pool.wait()
print('Writing GIF')
with multi_process_progressbar.MultiProcessProgressbar(len(svgs)) as progress:
    with imageio.get_writer(output_file, mode='I', duration=args.duration) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)
            progress.increment()
shutil.rmtree(temp_dir)
