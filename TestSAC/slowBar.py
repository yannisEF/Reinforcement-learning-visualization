from progress.bar import Bar

"""
    Progress bar printing the estimated time left
"""

class SlowBar(Bar):
    suffix = '%(remaining_time)s'
    @property
    def remaining_time(self):
        remain = self.eta
        affiche = ""
        if remain > 3600:
            hours = self.eta // 3600
            affiche += str(hours) + 'h '
            remain %= 3600
        if remain > 60:
            minutes = remain // 60
            affiche += str(minutes) + 'min '
            remain %= 60 
        else:
            affiche += '00min '
        affiche += str(remain) + 'sec'

        return affiche