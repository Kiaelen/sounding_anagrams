import torch


class ColorLView():
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        # Get L color by averaging color channels
        return torch.cat((2 * torch.stack([noise[:3].mean(0)] * 3), noise[3:]))

    def imprint(self, im):
        return self.inverse_view(im)

class ColorABView():
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        # Get AB color by taking residual
        return torch.cat((2 * (noise[:3] - torch.stack([noise[:3].mean(0)] * 3)), noise[3:]))
    
    def imprint(self, im):
        return self.inverse_view(im)

class LView_Composit():
    def __init__(self, cps_view):
        self.Lview = ColorLView()
        self.cps_view = cps_view
        
    def view(self, im):
        f, s = self.Lview, self.cps_view
        return s.view(f.view(im))
    
    def inverse_view(self, noise):
        f, s = self.Lview, self.cps_view
        return f.inverse_view(s.inverse_view(noise))
    
    def imprint(self, im):
        f, s = self.Lview, self.cps_view
        return f.inverse_view(im)
    
class ABView_Composit():
    def __init__(self, cps_view):
        self.ABview = ColorABView()
        self.cps_view = cps_view
        
    def view(self, im):
        f, s = self.ABview, self.cps_view
        return s.view(f.view(im))
    
    def inverse_view(self, noise):
        f, s = self.ABview, self.cps_view
        return f.inverse_view(s.inverse_view(noise))
    
    def imprint(self, im):
        f, s = self.ABview, self.cps_view
        return f.inverse_view(im)