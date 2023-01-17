from IPython.display import display, clear_output
import numpy as np
import time
import sys

from ..utils.operator import crossover, mutation, selection
from ..utils.load_data import Load

class AbstractModel:
    def __init__(self, seed=None, percent_print=0.5) -> None:
        self.seed = seed
        if seed is None:
            pass
        else:
            np.random.seed(seed)

        self.printed_before_percent = -2
        self.percent_print = percent_print 
        self.display_time = True 
        self.clear_output = True
    def compile(self, data_loc: str, crossover: crossover.AbstractCrossover, mutation: mutation.AbstractMutation, selection: selection.AbstractSelection):
        self.data = Load()
        self.data(data_loc)

        self.dim = self.data.num_bins + 1
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
    def fit(self, *args, **kwargs):
        self.time_begin = time.time()
        pass
    def render_process(self,curr_progress, list_desc, list_value, use_sys = False, *args, **kwargs): 
        percent = int(curr_progress * 100)
        if percent >= 100: 
            self.time_end = time.time() 
            percent = 100 
        else: 
            if percent - self.printed_before_percent >= self.percent_print:
                self.printed_before_percent = percent 
            else: 
                return 
                
        process_line = '%3s %% [%-20s]  '.format() % (percent, '=' * int(percent / 5) + ">")
        
        seconds = time.time() - self.time_begin  
        minutes = seconds // 60 
        seconds = seconds - minutes * 60 
        print_line = str("")
        if self.clear_output is True: 
            if use_sys is True: 
                # os.system("cls")
                pass
            else:
                clear_output(wait= True) 
        if self.display_time is True: 
            if use_sys is True: 
                # sys.stdout.write("\r" + "time: %02dm %.02fs  "%(minutes, seconds))
                # sys.stdout.write(process_line+ " ")
                print_line = print_line + "Time: %02dm %2.02fs "%(minutes, seconds) + " " +process_line
                
            else: 
                display("Time: %02dm %2.02fs "%(minutes, seconds))
                display(process_line)
        for i in range(len(list_desc)):
            desc = str("")
            if i < len(list_desc) - 1:
                desc = desc + f"{list_value[i]}" + " -"
            else:
                desc = desc + f"{list_value[i]}" + "    "
            line = '{}: {}  '.format(list_desc[i], desc)
            if use_sys is True: 
                print_line = print_line + line 
            else: 
                display(line)
        if use_sys is True: 
            # sys.stdout.write("\033[K")
            sys.stdout.flush() 
            sys.stdout.write("\r" + print_line)
            sys.stdout.flush()    
