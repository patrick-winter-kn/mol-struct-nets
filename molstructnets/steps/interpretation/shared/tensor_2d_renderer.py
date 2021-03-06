import gzip


def render(path, preprocessed, symbols, render_factor=50, show_grid=True, heatmap=None, background_heatmap=True):
    correction_x = 0.25 * render_factor
    correction_y = -0.125 * render_factor
    original_max_x = preprocessed.shape[0]
    original_max_y = preprocessed.shape[1]
    render_max_x = preprocessed.shape[0] * render_factor
    render_max_y = preprocessed.shape[1] * render_factor
    # Colorize background
    background = ''
    if heatmap is not None and background_heatmap:
        for x in range(original_max_x):
            for y in range(original_max_y):
                render_x = x * render_factor
                render_y = y * render_factor - render_factor
                pixel_size = str(render_factor * 1.01)
                color = '(' + str(heatmap[x, y, 0]) + ',' + str(heatmap[x, y, 1]) + ',' + str(heatmap[x, y, 2]) + ')'
                background += '<rect x="' + str(render_x) + '" y="' + str(render_y) + '" width="' + pixel_size \
                              + '" height="' + pixel_size + '" style="fill:rgb' + color + '" />\n'
    grid = ''
    if show_grid:
        grid += '<g>\n'
        # X grid
        for i in range(original_max_x + 1):
            x = i * render_factor
            grid += '<line x1="' + str(x) + '" y1="' + str(-render_factor) + '" x2="' + str(x) + '" y2="' + str(
                render_max_y - render_factor) + '" style="stroke:rgb(127,127,127);stroke-width:1" />\n'
        # Y grid
        for i in range(original_max_y + 1):
            y = i * render_factor - render_factor
            grid += '<line x1="' + str(0) + '" y1="' + str(y) + '" x2="' + str(render_max_x) + '" y2="' + str(
                y) + '" style="stroke:rgb(127,127,127);stroke-width:1" />\n'
        grid += '</g>\n'
    text = '<text x="0" y="0" fill="black" font-family="monospace" font-size="' + str(render_factor) + '">\n'
    for x in range(preprocessed.shape[0]):
        for y in range(preprocessed.shape[1]):
            symbol_index = preprocessed[x, y, :len(symbols)].argmax()
            if preprocessed[x, y, symbol_index] > 0:
                render_x = x * render_factor + correction_x
                render_y = y * render_factor + correction_y
                color = ''
                # Colorize characters
                if heatmap is not None:
                    if background_heatmap:
                        color = ' fill="rgb(255,255,255)" stroke="rgb(0,0,0)"'
                    else:
                        color = ' fill="rgb(' + str(heatmap[x, y, 0]) + ',' + str(heatmap[x, y, 1]) + ',' + \
                                str(heatmap[x, y, 2]) + ')"'
                text += '<tspan x="' + str(render_x) + '" y="' + str(render_y) + '"' + color + '>' + symbols[
                    symbol_index].decode('utf-8') + '</tspan>\n'
    text += '</text>\n'
    svg = '<svg viewBox="0 -' + str(render_factor) + ' ' + str(render_max_x) + ' ' + str(
        render_max_y) + '">\n' + background + grid + text + '</svg>\n'
    with gzip.open(path, 'wt') as file:
        file.write(svg)
