from cycler import cycler

class ObjDict(dict):
    '''
    This is an abstract class for allowing the keys of a dictionary to be accessed like class attributes.
    '''

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: "+name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: "+name)

## The following is the Paul Tol colourblind colour schemes for use in plots from https://personal.sron.nl/~pault/ ##

pt_bright = {
    "blue" : "#4477AA",
    "cyan" : "#66CCEE",
    "green" : "#228833",
    "yellow" : "#CCBB44",
    "red" : "#EE6677",
    "purple" : "#AA3377",
    "grey" : "#BBBBBB"
}

pt_hic = {
    "white" : "#FFFFFF",
    "yellow" : "#DDAA33",
    "red" : "#BB5566",
    "blue" : "#004488",
    "black" : "#000000"
}

pt_vibrant = {
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "teal" : "#009988",
    "orange" : "#EE7733",
    "red" : "#CC3311",
    "magenta" : "#EE3377",
    "grey" : "#BBBBBB"
}

pt_muted = {
    "indigo" : "#332288",
    "cyan" : "#88CCEE",
    "teal" : "#44AA99",
    "green" : "#117733",
    "olive" : "#999933",
    "sand" : "#DDCC77",
    "rose" : "#CC6677",
    "wine" : "#882255",
    "purple" : "#AA4499",
    "pale grey" : "#DDDDDD"
}

pt_bright_cycler = cycler(color=list(pt_bright.values()))

pt_hic_cycler = cycler(color=list(pt_hic.values()))

pt_vibrant_cycler = cycler(color=list(pt_vibrant.values()))

pt_muted_cycler = cycler(color=list(pt_muted.values()))