DIRECTIONS = {'up': '*',
              'down': '//',
              }


def sampling_calculation(value, nominator=2, sign='div'):
    assert sign in ['divide', 'div', '//', 'multiply', 'mult', '*']
    if sign in ['divide', 'div', '//']:
        return [v // nominator for v in value]
    elif sign in ['multiply', 'mult', '*']:
        return [v * nominator for v in value]