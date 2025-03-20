# 字符变布尔
def str2bool(x: str):
    if x == "False":
        return False
    elif x == "True":
        return True
    else:
        raise ValueError(
            'you should either input "True" or "False" but not {}'.format(x)
        )

# 把一个逗号分隔的数字字符串转换成一个整数列表
def list_of_ints(arg):
    return list(map(int, arg.split(",")))
