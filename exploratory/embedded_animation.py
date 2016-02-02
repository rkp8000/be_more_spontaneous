import io
import base64

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Mp4Animation(object):
    """
    Allows one to save an animation as a .mp4 file and embed the file into the output
    of an IPython notebook cell.
    """
    
    def make_animation(self, fname, fig, update_func, plot_objects, data, n_frames, interval,
                       writer='ffmpeg', fps=15, bitrate=1800, blit=True):
        """
        Create an animation and save it.
        """
        Writer = animation.writers[writer]
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
        
        anim = animation.FuncAnimation(
            fig, update_func, n_frames, fargs=(plot_objects, data), interval=interval, blit=blit
        )
        
        anim.save(fname, writer=writer)
        
        plt.close()
        self.fname = fname
        
    def get_animation_html(self):
        video = io.open(self.fname, 'r+b').read()
        encoded = base64.b64encode(video)
        html = '''<video alt="test" controls> <source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''.format(encoded.decode('ascii'))
        return htm