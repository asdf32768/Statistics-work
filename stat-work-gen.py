# from handcalcs.decorator import handcalc
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import latexify
from math import sqrt
import scipy
from fractions import Fraction as frac
# import prettytable

from pylatex import (
    Alignat,
    Axis,
    Document,
    Figure,
    Math,
    Matrix,
    Plot,
    Section,
    Subsection,
    Tabu,
    TikZ,
    NoEscape,
    Package,
    Command,
    TextBlock,
    MultiColumn,
    HugeText,
    PageStyle,
    Head,
    NewPage,
    MiniPage,
    Alignat,
)
from pylatex.section import Paragraph
from pylatex.utils import rm_temp_dir, bold, italic
from pylatex.basic import NewLine, LineBreak
from pylatex.position import HorizontalSpace, VerticalSpace, Center, FlushLeft, FlushRight
from pylatex.config import active
from pylatex.base_classes import Environment
import argparse

GLOBAL_DATA = {}

@latexify.expression
def COUNT():
    return sum((n[i]) for i in range(1, k+1))

@latexify.expression
def MEAN():
    return sum((x[i]*n[i]) for i in range(1,k+1))/n

@latexify.expression
def D_B():
    return sum(((x[i]**2)*n[i]) for i in range(1, k+1))/n - (x[B]**_)**2

@latexify.expression(use_math_symbols=True)
def sigma_B():
    return sqrt(D[B](x))

@latexify.expression(use_math_symbols=True)
def U_3():
    return 1/(n*sigma[B]**3)*(sum(n[i]*(x[i]-x[B]**_)**3 for i in range(1, k+1)))

@latexify.expression(use_math_symbols=True)
def U_4():
    return 1/(n*sigma[B]**4)*(sum(n[i]*(x[i]-x[B]**_)**4 for i in range(1, k+1))) - 3

@latexify.expression(use_math_symbols=True)
def ChiSquared():
    return chi[HOPM]**2 == sum(((n[i]-m[i])**2)/m[i] for i in range(1, k+1))

@latexify.expression(use_math_symbols=True)
def ChiSquaredEq():
    return chi[PABH]**2 == sum(((n[i]-m[i])**2)/m[i] for i in range(1, k+1))

@latexify.expression(use_math_symbols=True)
def Sum5():
    return sum((x[i]-alpha)**2 for i in range(1, n+1))

def calc_x_b(A):
    return np.round(np.dot(A[0, :], A[1, :]) / np.sum(A[1, :]), 4).item()

def calc_d_b(A):
    return np.round((np.dot((A[0,:]**2),A[1,:]) / np.sum(A[1,:])) - (calc_x_b(A)**2), 4).item()

def calc_o_b(A):
    return np.round(np.sqrt(calc_d_b(A)), 4).item()

def calc_u_3(A):
    return np.round(np.dot((A[0, :] - calc_x_b(A))**3, A[1, :]) / (np.sum(A[1, :]) * (calc_o_b(A)**3)), 4).item()

def calc_u_4(A):
    return np.round(np.dot((A[0, :] - calc_x_b(A))**4, A[1, :]) / (np.sum(A[1, :]) * (calc_o_b(A)**4)), 4).item()

def clamp(n, a, b): 
    if n < a: 
        return a
    elif n > b: 
        return b
    else: 
        return n

def interval_intersect(a,b):
    a0,a1 = a
    b0,b1 = b
    return max(0,min(a1,b1)-max(a0,b0))

def calc_interval(interval:list, a, b):
    if interval[0] >= a and interval[1] <= b: return (interval[1], interval[0], np.round(abs(interval[1] - interval[0]), 4).item())
    if bool(interval_intersect(interval, [a, b])):
        return (clamp(interval[1], a, b), clamp(interval[0], a, b), np.round(abs(clamp(interval[1], a, b) - clamp(interval[0], a, b)), 4).item())
    else:
        return (0, 0, 0)

def round_expr(expr, num_digits):
     return expr.xreplace({n.evalf() : n if type(n)==int else sp.Float(n, num_digits) for n in expr.atoms(sp.Number)})

def expand_expr_new(Expression:str):
    expr = sp.S(Expression, evaluate=False)
    # expr = round_expr(expr, 4)
    return (sp.latex(expr), np.round(eval(str(expr)), 4).item())

def expand_expr(Expression:str,numbers:list):
    x = sp.symbols('x0:{}'.format(len(numbers)))
    symbol_array = []

    for i in range(len(numbers)):
        symbol_array.append(x[i])
    
    expr = Expression.format(*symbol_array)
    for i in range(len(numbers)-1,-1,-1):
        expr = expr.replace('x{}'.format(i), 'x[{}]'.format(i))
    
    expr = eval(expr)

    output = 0

    with sp.evaluate(False):
        subs = []
        for i in range(len(numbers)):
            subs.append((x[i], numbers[i]))
        expr_subs = expr.subs(subs)
        expr_subs = round_expr(expr_subs, 4)
        res = np.round(eval(str(expr_subs)), 4).item()
        # res = np.round(float(expr_subs.doit()), 4).item()
        output = (expr_subs, sp.latex(expr_subs), res)
        
    return output

def expand_sum(Expression:str, arrays:list):
    max_length = sum(len(i) for i in arrays)
    x = sp.symbols('x0:{}'.format(max_length))
    symbol_arrays = []
    index = 0
    for arr in arrays:
        symbol_arrays.append([x[_] for _ in range(index, index + len(arr))])
        index += len(arr)
    expr = Expression.format(*['np.array({})'.format(str(i).replace('(','[').replace(')',']')) for i in symbol_arrays])
    for i in range(max_length-1, -1, -1):
        expr = expr.replace('x{}'.format(i), 'x[{}]'.format(i))
    expr = eval(expr)
    with sp.evaluate(False):
        subs = []
        index = 0
        for i in range(len(arrays)):
            for j in range(len(arrays[i])):
                subs.append((x[index], float(arrays[i][j])))
                index += 1
        expr_subs = expr.subs(subs)
        expr_subs = round_expr(expr_subs, 4)
    return (expr_subs, sp.latex(expr_subs), np.round(float(expr_subs.doit()), 4).item())

def cdf(x):
    """
    Расчет функции Лапласа.

    Внимание! Отключена проверка по границам
    """

    x = x / 1.414213562
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    s = np.sign(x)
    t = 1 / (1 + s * p * x)
    b = np.exp(-x * x)
    y = (s * s + s) / 2 - \
        s * (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * b / 2
    return y - 0.5

def generate_header(doc):
    global GLOBAL_DATA
    header = PageStyle('header')
    with header.create(Head("R")) as h:
        with header.create(Tabu('|c',width=1)) as table:
            table.add_row((GLOBAL_DATA['ФИО'],),strict=False)
            table.add_row((GLOBAL_DATA['Группа'],),strict=False)
            table.add_row((NoEscape(r'$m={}$ $n={}$'.format(GLOBAL_DATA['m'], GLOBAL_DATA['n'])),),strict=False)
            table.add_row((NoEscape(r'$m_1={}$ $n_1={}$'.format(GLOBAL_DATA['m1'], GLOBAL_DATA['n1'])),),strict=False)
            # table.add_row(('Работа №{}'.format(WORK_NUMBER),),strict=False)
            table.add_row((NoEscape(r'Лист №\thepage'),))
            table.add_hline()
    doc.preamble.append(header)
    doc.change_document_style('header')

def generate_random_var(doc, var, names):
    assert((var.shape[0] == len(names)) or len(names) == 0)
    tspec = '|'+'|'.join('c'*var.shape[1])+'|'
    if len(names) > 0:  tspec = tspec + 'c|'
    with doc.create(Tabu(tspec,spread='15mm')) as table:
        for i in range(var.shape[0]):
            if len(names) == 0:
                table.add_hline()
                table.add_row(list(var[i, :]))
            else:
                table.add_hline()
                # row = np.append(NoEscape(names[i]), var[i].astype(str))
                row = [NoEscape(names[i])] + list(var[i, :])
                table.add_row(row)
        table.add_hline()

def format_math_str(s):
    return s.replace('i = 1', 'REPLACE').replace(' = ', '$ = $').replace('REPLACE', 'i = 1')

def generate_sum(start, end, operator):
    return r'\sum_{{{}}}^{{{}}}{}'.format(start, end, operator)

def generate_fraction(start, end):
    return r'\frac{{{}}}{{{}}}'.format(start, end)


def main():
    global GLOBAL_DATA
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True, help="ФИО")
    parser.add_argument("--group", type=str, required=True, help="Группа")

    parser.add_argument("-m", "--m", type=int, required=True, help="первая m")
    parser.add_argument("-n", "--n", type=int, required=True, help="первая n")
    parser.add_argument("-m1", "--m1", type=int, required=True, help="вторая m")
    parser.add_argument("-n1", "--n1", type=int, required=True, help="вторая n")
    parser.add_argument("--pdf", action='store_true', help="Сгенерировать pdf (Да: --pdf /Нет: )")
    parser.add_argument("--tex", action='store_true', help="Сгенерировать tex (Да: --tex /Нет: )")

    args = parser.parse_args()

    bPdf = bool(args.pdf)
    bTex = bool(args.tex)

    GLOBAL_DATA = {'ФИО': str(args.name),'Группа':str(args.group),'m':int(args.m),'n':int(args.n), 'm1': int(args.m1), 'n1': int(args.n1)}
    NP_DATA_VAR, SP_DATA_VAR, DATA_NUMBERS, INTERVAL_VAR, INTERVALS, INTERVAL_VAR_NEW, DATA_VAR_FIRST, P_NORM, P_EQ = [None] * 9
    X_VALS, S_VALS, F_VALS, ARRAYS = [None] * 4

    latexes_of_vars = {'x_b': r'\overline{x}_B', 'd_b': 'D_B(x)', 'sigma_b': r'\sigma_B(x)', 'u_3': 'A_B', 'u_4': 'E_B'}

    count, mean, d_b, o_b, a_b_u_3, e_b_u_4, chi_squared, chi_squared_eq, chi_squared_eq_mmp = [0] * 9
    rm_temp_dir()

    geometry_options = {"margin": "0.7in", 'tmargin':'20mm', 'bmargin':'20mm', 'lmargin':'25mm', 'rmargin':'25mm'}

    doc = Document(documentclass='article',\
                   document_options=None,\
                   fontenc=['T2A', 'T1'],\
                     lmodern=None,\
                     textcomp=None,\
                     geometry_options=geometry_options,\
                     page_numbers=None,\
                     indent=False,\
                     data=None,\
                     font_size='large')

    doc.change_length('\TPHorizModule', '1mm')
    doc.change_length('\TPVertModule', '1mm')

    doc.packages.add(Package('geometry'))
    doc.packages.add(Package('caption'))
    # doc.packages.add(Package('pgf'))
    doc.packages.add(Package('textpos',options=['absolute','overlay']))
    doc.packages.add(Package('hyphsubst'))
    doc.packages.add(Package('babel', options=['english', 'main=russian']))
    doc.packages.add(Package('newtxtext'))
    doc.packages.add(Package('newtxmath'))
    doc.packages.add(Package('ragged2e'))
    doc.preamble.append(NoEscape(r'\DeclareFontFamilySubstitution{T2A}{\familydefault}{Tempora-TLF}'))
    doc.packages.add(Package('lastpage'))
    doc.packages.add(Package('indentfirst'))
    doc.preamble.append(Command(r'linespread',arguments='1.5'))
    doc.preamble.append(NoEscape(r'\setlength{\parindent}{5ex}'))
    doc.preamble.append(NoEscape(r'\setlength{\parskip}{1ex}'))
    doc.append(NoEscape(r'\pretolerance=10000'))
    doc.append(NoEscape(r'\fontsize{12}{12pt}\selectfont'))
    doc.append(NoEscape(r'\RaggedRight'))
    doc.append(NoEscape(r'\setlength{\headsep}{40pt}'))

    generate_header(doc)

    # doc.append(Command('title', 'Самостоятельные работы по статистике'))
    # doc.append(Command('author', GLOBAL_DATA['ФИО']))
    # doc.append(Command('date', NoEscape(r'\today')))
    # doc.append(NoEscape(r'\maketitle'))


    m = GLOBAL_DATA['m']
    n = GLOBAL_DATA['n']

    data_1 = np.array([[m-2	,n	,m-1	,n+1	,n-1],\
                      [m+1	,m+n	,n	,m+2	,m+1],\
                      [n-2	,m	,n-1	,m	,n+1	],\
                      [m-2	,n*2	,m	,n	,m-1	],\
                      ['' 	  ,m	,m+2	,n+2,n-1]])

    with doc.create(Section(HugeText('Работа №1'),numbering=False,label=False)) as sec:
        # with sec.create(MiniPage(align='left')):
        sec.append('Данные')
        sec.append(LineBreak())
        sec.append(NewLine())
        with sec.create(Tabu('|'+' '.join('c'*data_1.shape[0])+'|',width=data_1.shape[0],spread='15mm')) as table:
            table.add_hline()
            for i in range(data_1.shape[0]):
                table.add_row(data_1[i])
            table.add_hline()
        sec.append(LineBreak())
        sec.append(NewLine())
        
        data_1[4, 0] = '-1'
        data_1 = data_1.astype(int)

        data_1 = data_1.flatten(order='C')
        data_1 = np.delete(data_1, np.where(data_1 == -1))
        data_1.sort()

        sec.append('Порядковая статистика')
        sec.append(LineBreak())
        sec.append(NewLine())

        with sec.create(Tabu(' '.join('c'*data_1.size),width=data_1.size,spread='5mm')) as table:
            table.add_row(data_1)
        sec.append(LineBreak())
        sec.append(NewLine())

        # Changes
        ## data_1 = np.array([1] + [2] * 3 + [4] * 6 + [5] * 2 + [6] * 3 + [7] * 4 + [8] * 3 + [9] * 2 + [10])

        data_info = np.unique(data_1, return_counts=True)
        data_var = np.zeros((2, data_info[0].size),dtype=int)
        data_var[0, :] = data_info[0]
        data_var[1, :] = data_info[1]

        DATA_VAR_FIRST = data_var.copy()
        # data_var[2, :] = np.array(np.round(data_var[1, :] / np.sum(data_var[1, :]), 4), dtype=float)

        count = np.sum(data_var[1, :])
        mean = np.round(np.mean(data_1),4)
        d_b = np.round((np.dot((data_var[0,:]**2),data_var[1,:]) / count) - (mean**2), 4)
        o_b = np.round(np.sqrt(d_b), 4)
        a_b_u_3 = np.round(np.dot(((data_var[0, :]-mean)**3),data_var[1,:]) / (count * (o_b**3)), 4)
        e_b_u_4 = np.round(np.dot(((data_var[0, :]-mean)**4),data_var[1,:]) / (count * (o_b**2)) - 3, 4)


        sp_data_var = sp.Matrix(data_var.shape[0]+1, data_var.shape[1], lambda i, j: sp.S(data_var[i, j]) if i != 2 else sp.S(data_var[i-1, j]) / count)

        generate_random_var(sec, sp_data_var, [r'$x_i$', r'$n_i$', r'$w_i$'])
        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(r'Количество = ${}$=${}$'.format(format_math_str(COUNT._latex), count)))
        sec.append(LineBreak())
        sec.append(NewLine())
        if data_1.size % 2 != 0:
            sec.append(NoEscape(r'Медиана = {}'.format(np.median(data_1))))
        else:
            sec.append(NoEscape(r'Медиана = $\frac{{{}+{}}}{{2}}$ = ${}$'.format(data_1[(data_1.size//2)-1], data_1[data_1.size//2], np.median(data_1))))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'Мода = ${}$'.format(data_var[0][np.argmax(data_var[1])])))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'Среднее = ${}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['x_b'], format_math_str(MEAN._latex), expand_sum('np.sum({{}})/{}'.format(count), [data_1])[1], mean)))
        # sec.append(NoEscape(r'Среднее = $\overline{{x}} = {} = {} = {}$'.format(generate_fraction('1','n') + generate_sum('i=1','k','x_in_i'), generate_fraction(expand_sum(r'{}', [data_1]), str(count)), mean)))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'${}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['d_b'], format_math_str(D_B._latex), expand_sum('1/{}*np.dot({{}}**2, {{}})-{}**2'.format(count, mean), [data_var[0,:], data_var[1,:]])[1], d_b)))
        # sec.append(NoEscape(r'$D_B(x) = {} = {} = {}$'.format(generate_fraction('1','n') + generate_sum('i=1','k','x_i^2n_i'), generate_fraction(expand_sum(r'{}', [data_1]), str(count)), d_b)))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'${}$ = ${}$ = $\sqrt{{{}}}$ = ${}$'.format(latexes_of_vars['sigma_b'], format_math_str(sigma_B._latex), d_b, o_b)))
        sec.append(LineBreak())
        sec.append(NewLine())
        # sec.append(NoEscape)
        # Expression = "np.dot({}**2,{})/5"
        # expand_sum(Expression=Expression, arrays=[np.array([1,2,3]),np.array([4,5,6])])
        sec.append(NoEscape(r'\begin{equation}'))
        sec.append(NoEscape(r'F_n(x) = '))
        sec.append(NoEscape(r'\begin{cases}'))
        sec.append(NoEscape(r'0, & \text{{если}}\ x \leq {} \\'.format(data_var[0, 0])))
        for i in range(data_var.shape[1]-1):
            sec.append(NoEscape(r'{}, & \text{{если}}\ {} < x \leq {} \\'.format(sp.S(np.sum(data_var[1, :(i+1)])) / count, data_var[0, i], data_var[0, i+1])))
        sec.append(NoEscape(r'1, & \text{{если}}\ x > {}'.format(data_var[0,-1])))
        sec.append(NoEscape(r'\end{cases}'))
        sec.append(NoEscape(r'\end{equation}'))
        sec.append(LineBreak())
        sec.append(NewLine())

        NP_DATA_VAR = data_var
        SP_DATA_VAR = sp_data_var
        DATA_NUMBERS = data_1

    with doc.create(Section(HugeText('Работа №2'),numbering=False,label=False)) as sec:
        sec.append('Данные')
        sec.append(LineBreak())
        sec.append(NewLine())
        data_var = NP_DATA_VAR
        sp_data_var = SP_DATA_VAR
        generate_random_var(sec, sp_data_var, [r'$x_i$', r'$n_i$', r'$w_i$'])
        sec.append(LineBreak())
        sec.append(NewLine())


        A_B, res_a_b = expand_sum('np.dot(({{}} - {})**3, {{}})/({}*({}**3))'.format(mean, count, o_b), [data_var[0, :], data_var[1, :]])[1:3]
        E_B, res_e_b = expand_sum('np.dot(({{}} - {})**4, {{}})/({}*({}**4))-3'.format(mean, count, o_b), [data_var[0, :], data_var[1, :]])[1:3]

        a_b_u_3 = np.round(float(res_a_b), 4).item()
        e_b_u_4 = np.round(float(res_e_b), 4).item()

        with sec.create(Subsection('График вариационного ряда',numbering=False,label=False)):
            with sec.create(TikZ()):
                ytick = ', '.join(str(i.item()) for i in data_var[1, :])
                xtick = ', '.join(str(i.item()) for i in data_var[0, :])
                # plot_options = NoEscape(r'height=8cm, width=12cm, grid=major, ytick={{0,1,...,{}}}, xtick={{0,1,...,{}}}'.format(np.max(data_var[1, :]), np.max(data_var[0, :])))
                plot_options = NoEscape(r'height=8cm, width=12cm, grid=major, ytick={{{}}}, xtick={{{}}}'.format(ytick, xtick))
                # plot_options = NoEscape(r'height=4cm, width=6cm, grid=major, ytick={0,1,...,' + str(np.max(data_var[1, :])) + r'}, xtick={0,1,...,' + str(np.max(data_var[0, :])) + r'}')
                with sec.create(Axis(options=plot_options)) as plot:
                    plot.append(Plot(name='Полигон', coordinates=data_var.T))
        
        sec.append(NoEscape(r'Выборочный коэффициент ассиметрии\\'))
        # sec.append(NoEscape(r'$\frac{1}{n \cdot {\sigma_B}^3}\sum_{i=1}{k}{(n_i \cdot {(x_i - \overline{x}_B)}^3)} = $'))
        sec.append(NoEscape(r'${}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['u_3'], format_math_str(U_3._latex), A_B, res_a_b)))
        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(r'Выборочный коэффициент эксцесса\\'))
        sec.append(NoEscape(r'${}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['u_4'], format_math_str(U_4._latex), E_B, res_e_b)))
        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(r'Составим интервальный вариационный ряд\\'))
        k = np.round(1 + 3.32 * np.log10(count)).item()
        min_ = np.min(data_var[0, :]).item()
        max_ = np.max(data_var[0, :]).item()
        R = max_ - min_
        delta = np.round(R/(k-1), 1).item()
        x = [np.round(min_ - delta / 2, 4).item()]
        sec.append(NoEscape(fr'Число интервалов для $n$ = ${count}$ по формуле Стерджесса:\\$k$ = $1 + 3.32 \cdot \lg{{n}}$ = $1 + 3.32 \cdot \lg{count}$ = ${1 + 3.32 * np.log10(count)} \approx {k}$\\'))
        sec.append(NoEscape(fr'Вариационный размах $R$ = $x_{{\max }} - x_{{\min }}$ = ${max_} - {min_}$ = ${R}$\\'))
        sec.append(NoEscape(fr'Величина интервала: $\Delta = h$  = $R \over {{k - 1}}$ = ${R} \over {k - 1}$ = ${R/(k-1)} \approx {delta}$\\'))
        sec.append(NoEscape(r'Находим границы интервалов: \\'))
        sec.append(NoEscape(fr'$x_{{1}}$ = $x_{{\min}} - 0.5 \cdot h$  = ${min_} - 0.5 \cdot {delta}$ = ${x[0]}$\\'))

        for i in range(1, int((max_-x[0])/delta) + 2):
            sec.append(NoEscape(fr'$x_{{{i+1}}}$ = $x_{{{i}}} + h$ = ${np.round(x[0] + delta * i, 4).item()}$\\'))
            x.append(np.round(x[0] + delta * i, 4).item())
        
        INTERVALS = np.array(x)
        x = [[x[i], x[i+1]] for i in range(len(x)-1)]

        interval_var = np.empty((2, len(x)), dtype=object)
        interval_var[0, :] = [str(i).replace(']',')') for i in x]
        x_vals = [0] * len(x)

        for i in range(interval_var.shape[1]):
            x_vals[i] = np.sum((x[i][0] <= DATA_NUMBERS) & (DATA_NUMBERS < x[i][1]))

        interval_var[1, :] = x_vals

        coords = np.array([[np.round((i[0] + i[1]) / 2, 4).item() for i in x], x_vals])

        INTERVAL_VAR = coords.copy()

        coords_copy = coords.copy().T

        sec.append(VerticalSpace('10pt'))
        
        generate_random_var(sec, interval_var, [r'$I_i$', r'$n_i$'])

        sec.append(VerticalSpace('10pt'))

        with sec.create(Subsection('График интервального вариационного ряда',numbering=False,label=False)):
            with sec.create(TikZ()):
                plot_options = NoEscape(r'height=8cm, width=12cm, grid=both, ytick={{{}}}, xtick={{{}}}'.format(', '.join(str(i.item()) for i in coords[1, :]), ', '.join(str(i.item()) for i in coords[0, :])))
                # plot_options = NoEscape(r'height=4cm, width=6cm, grid=major, ytick={0,1,...,' + str(np.max(data_var[1, :])) + r'}, xtick={0,1,...,' + str(np.max(data_var[0, :])) + r'}')
                with sec.create(Axis(options=plot_options)) as plot:
                    plot.append(Plot(name='График', coordinates=coords_copy, options='blue'))
        
        sec.append(VerticalSpace('10pt'))

        coords[0, :] = [i[0] for i in x]

        coords = np.concatenate((coords, np.array([[x[-1][1], x_vals[-1]]]).T), axis=1)

        with sec.create(Subsection('Гистограмма интервального вариационного ряда',numbering=False,label=False)):
            with sec.create(TikZ()):
                plot_options = NoEscape(r'height=8cm, width=12cm, area style, ytick={{{}}}, xtick={{{}}}'.format(', '.join(str(i.item()) for i in coords[1, :]), ', '.join(str(i.item()) for i in coords[0, :])))
                # plot_options = NoEscape(r'height=4cm, width=6cm, grid=major, ytick={0,1,...,' + str(np.max(data_var[1, :])) + r'}, xtick={0,1,...,' + str(np.max(data_var[0, :])) + r'}')
                with sec.create(Axis(options=plot_options)) as plot:
                    plot.append(Plot(name='Гистограмма', coordinates=coords.T,options=['ybar interval', 'blue']))
        
    with doc.create(Section(HugeText('Работа №3'),numbering=False,label=False)) as sec:
        data_var = INTERVAL_VAR

        generate_random_var(sec, data_var, [r'$I_i$', r'$n_i$'])

        sec.append(LineBreak())
        sec.append(NewLine())

        mean_latex, mean = expand_sum('np.dot({{}}, {{}})/{}'.format(count), [data_var[0, :], data_var[1, :]])[1:3]
        mean = np.round(float(mean), 4).item()
        d_b_latex, d_b = expand_sum('1/{}*np.dot({{}}**2, {{}})-{}**2'.format(count, mean), [data_var[0,:], data_var[1,:]])[1:3]
        d_b = np.round(float(d_b), 4).item()
        o_b = np.round(np.sqrt(d_b), 4).item()


        sec.append(NoEscape(r'${}^{{*}}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['x_b'], format_math_str(MEAN._latex), mean_latex, mean)))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'${}^{{*}}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['d_b'], format_math_str(D_B._latex), d_b_latex, d_b)))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'${}^{{*}}$ = ${}$ = $\sqrt{{{}}}$ = ${}$'.format(latexes_of_vars['sigma_b'], format_math_str(sigma_B._latex), d_b, o_b)))
        sec.append(LineBreak())
        sec.append(NewLine())
        A_B, res_a_b = expand_sum('np.dot(({{}} - {})**3, {{}})/({}*({}**3))'.format(mean, count, o_b), [data_var[0, :], data_var[1, :]])[1:3]
        E_B, res_e_b = expand_sum('np.dot(({{}} - {})**4, {{}})/({}*({}**4))-3'.format(mean, count, o_b), [data_var[0, :], data_var[1, :]])[1:3]
        a_b_u_3 = np.round(float(res_a_b), 4).item()
        e_b_u_4 = np.round(float(res_e_b), 4).item()
        sec.append(NoEscape(r'${}^{{*}}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['u_3'], format_math_str(U_3._latex), A_B, res_a_b)))
        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NoEscape(r'${}^{{*}}$ = ${}$ = ${}$ = ${}$'.format(latexes_of_vars['u_4'], format_math_str(U_4._latex), E_B, res_e_b)))
        sec.append(LineBreak())
        sec.append(NewLine())

        laplaceArr = []
        laplaceArr1 = []
        for i in INTERVALS:
            val1 = np.round((i-mean)/o_b, 4)
            val2 = np.round(cdf((i-mean)/o_b), 4)
            sec.append(NoEscape(r"Ф$\left(\frac{{{} - {}}}{{{}}}\right)$ = Ф$({})$ = ${}$\\".format(i.item(), float(mean), float(o_b), val1.item(), val2.item())))
            laplaceArr.append(val2.item())
        sec.append(VerticalSpace('10pt'))
        for i in range(len(INTERVALS) - 1):
            sec.append(NoEscape(r"Ф$({} < x < {})$ = ${} -{}$ = ${}$\\".format(INTERVALS[i].item(), INTERVALS[i+1].item(), laplaceArr[i+1], laplaceArr[i], np.round(laplaceArr[i+1] - laplaceArr[i], 4).item()).replace('--', '+ ')))
            laplaceArr1.append(np.round(laplaceArr[i+1] - laplaceArr[i], 4).item())
        sec.append(VerticalSpace('10pt'))

        P_NORM = np.array(laplaceArr1)

        for i, k in enumerate(laplaceArr1):
            sec.append(NoEscape(r"$m_{}$ = ${} \cdot {} $= ${}$\\".format(i+1, k, count.item(), np.round(count*k, 4).item())))
            laplaceArr1[i] = np.round(count * k, 4).item()
        sec.append(VerticalSpace('10pt'))

        data_var = np.concatenate((data_var, np.array([laplaceArr1])), axis=0)

        generate_random_var(sec, data_var, [r'$I_i$', r'$n_i$', r'$m_i$'])

        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(VerticalSpace('10pt'))

        chi_squared_expr = expand_sum(r"np.dot(({}-{})**2,1/{})",[data_var[1, :], data_var[2, :], data_var[2, :]])

        chi_squared = np.round(float(chi_squared_expr[2]), 4).item()

        sec.append(NoEscape(fr'${format_math_str(ChiSquared._latex)}$ = ${chi_squared_expr[1]}$ = ${chi_squared}$'))

        INTERVAL_VAR = data_var

        sec.append(LineBreak())
        sec.append(NewLine())

    with doc.create(Section(HugeText('Работа №4'),numbering=False,label=False)) as sec:
        data_var = INTERVAL_VAR

        a_hat, b_hat = [np.round(i, 4).item() for i in (mean - np.sqrt(3) * o_b, mean + np.sqrt(3) * o_b)]
        f_x = np.round(1/(b_hat-a_hat), 4).item()
        sec.append(NoEscape(r'$\hat{\alpha}=\bar{x}_{B}-\sqrt{3}\cdot \sigma_{B}(x)=$'))
        sec.append(NoEscape(fr'${mean}-\sqrt{{3}}\cdot {o_b} \approx {a_hat}$\\'))
        sec.append(NoEscape(r'$\hat{\beta}=\bar{x}_{B}+\sqrt{3}\cdot \sigma_{B}(x)=$'))
        sec.append(NoEscape(fr'${mean}+\sqrt{{3}}\cdot {o_b} \approx {b_hat}$\\'))

        sec.append(VerticalSpace('10pt'))

        sec.append(NoEscape(r'''$f(x) = \left\{ \begin{array}{cl}
    0 &,\ x < \hat{\alpha} \\
    \frac{1}{\hat{\beta}-\hat{\alpha}} &,\hat{\alpha}\le x \le \hat{\beta} \\
    0 &, \ x > \hat{\beta}
    \end{array} \right.'''))
        
        sec.append(NoEscape(fr'''=\left\{'{'} \begin{{array}}{{cl}}
    0 &,\ x < {a_hat} \\
    {f_x} &,{a_hat}\le x \le {b_hat} \\
    0 &, \ x > {b_hat}
    \end{{array}} \right.$\\'''))

        sec.append(VerticalSpace('10pt'))

        x = INTERVALS
        x = [[x[i], x[i+1]] for i in range(len(x)-1)]

        p = np.zeros(len(x))

        for i in range(len(p)):
            begin, end, p[i] = calc_interval(x[i], a_hat, b_hat)
            p[i] *= f_x
            p[i] = np.round(p[i], 4)
            sec.append(NoEscape(fr'$p_{{{i+1}}}=({begin}-{end})\cdot f(x)=$'))
            sec.append(NoEscape(fr'${np.round(begin-end, 4).item()}\cdot {f_x} = {p[i].item()}$\\'))

        P_EQ = np.array(p)

        # m_1 = np.round(count * p[0], 4).item()
        # m_2 = np.round(count * p[1], 4).item()
        m = np.round(p * count, 4)

        sec.append(VerticalSpace('10pt'))

        for i in range(len(m)):
            sec.append(NoEscape(fr'$m_{{{i+1}}} = n \cdot p_{{{i+1}}} = {count} \cdot {p[i]} = {m[i].item()}$\\'))

        data_var[2, :] = m

        INTERVAL_VAR_NEW = data_var

        zeros_m = np.where(m==0)

        sec.append(VerticalSpace('10pt'))

        generate_random_var(sec, data_var, [r'$I_i$', r'$n_i$', r'$m_i$'])

        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(VerticalSpace('10pt'))

        chi_squared_eq_expr = expand_sum(r"np.dot(({}-{})**2,1/{})",[np.delete(data_var[1, :], zeros_m), np.delete(data_var[2, :], zeros_m), np.delete(data_var[2, :], zeros_m)])

        chi_squared_eq = np.round(float(chi_squared_eq_expr[2]), 4).item()

        sec.append(NoEscape(fr'${format_math_str(ChiSquaredEq._latex)}$ = ${chi_squared_eq_expr[1]}$ = ${chi_squared_eq}$'))

        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(r'''С учетом вида гистограммы, значений $A_B$ и $E_B$ \\
                             и стат. $\chi^2$ для равн. и норм. распределения делаем вывод:\\
                            $A_B$ близок к нулю $\implies$ плотность симметрично $E^*_B$ не особо отличается от $E_B$;\\
                            $\chi^2$ не велики, т.к. малый объем выборки и мало интервалов.'''))

        sec.append(LineBreak())
        sec.append(NewLine())

    with doc.create(Section(HugeText('Работа №5'),numbering=False,label=False)) as sec:
        sec.append(NoEscape(r'$\overline{{x}}_B$ = ${}$\\'.format(mean)))
        sec.append(NoEscape(r'$D_B(x)$ = ${}$\\'.format(d_b)))
        sec.append(NoEscape(r'$\sigma_B(x)$ = ${}$\\'.format(o_b)))

        gammas = [0.9, 0.95]

        for i in range(len(gammas)):
            sec.append(VerticalSpace('10pt'))
            c_gamma = np.round(scipy.stats.norm.ppf((1 + gammas[i]) / 2), 4).item()
            a, b = [np.round(i, 4).item() for i in (mean - c_gamma * o_b / np.sqrt(count), mean + c_gamma * o_b / np.sqrt(count))]
            sec.append(NoEscape(fr'{i+1}) $\gamma = {gammas[i]}\qquad \gamma/2 = {np.round(gammas[i] / 2, 4).item()} \implies C_{{\gamma}} = {c_gamma}$ (из таблицы)\\'))
            sec.append(NoEscape(fr'$\overline{{x}}_B-C_{{\gamma}}\cdot \frac{{\sigma}}{{\sqrt{{n}}}} = {mean} - {c_gamma}\cdot \frac{{{o_b}}}{{{np.round(np.sqrt(count), 4).item()}}} = {a}$\\'))
            sec.append(NoEscape(fr'$\overline{{x}}_B+C_{{\gamma}}\cdot \frac{{\sigma}}{{\sqrt{{n}}}} = {mean} + {c_gamma}\cdot \frac{{{o_b}}}{{{np.round(np.sqrt(count), 4).item()}}} = {b}$\\'))
            sec.append(NoEscape(fr'$P({a} < \alpha < {b}) = {gammas[i]}$\\'))

        sec.append(VerticalSpace('10pt'))

        S = np.round(np.sqrt((d_b*count)/(count-1)), 4).item()
        df = count - 1

        sec.append(NoEscape(fr'$S = \sqrt{{\frac{{n}}{{n-1}}\cdot D_B(x)}} = \sqrt{{\frac{{{count}}}{{{count-1}}}\cdot {d_b}}} \approx {S}$\\'))
        sec.append(NoEscape(fr'Количество степеней свободы = $df = n - 1 = {count} - 1 = {df}$\\'))

        for i in range(len(gammas)):
            sec.append(VerticalSpace('10pt'))
            t_gamma = np.round(scipy.stats.t.ppf((1 + gammas[i]) / 2, df), 4).item()
            a, b = [np.round(i, 4).item() for i in (mean - t_gamma * S / np.sqrt(count), mean + t_gamma * S / np.sqrt(count))]
            sec.append(NoEscape(fr'{i+3}) $\gamma = {gammas[i]}$\\'))
            sec.append(NoEscape(fr'$t_{{\gamma}} = {t_gamma}$ (из таблицы)\\'))
            sec.append(NoEscape(fr'$\overline{{x}}_B-t_{{\gamma}}\cdot \frac{{S}}{{\sqrt{{n}}}} = {mean} - {t_gamma}\cdot \frac{{{S}}}{{{np.round(np.sqrt(count), 4).item()}}} = {a}$\\'))
            sec.append(NoEscape(fr'$\overline{{x}}_B+t_{{\gamma}}\cdot \frac{{S}}{{\sqrt{{n}}}} = {mean} + {t_gamma}\cdot \frac{{{S}}}{{{np.round(np.sqrt(count), 4).item()}}} = {b}$\\'))
            sec.append(NoEscape(fr'$P({a} < \alpha < {b}) = {gammas[i]}$\\'))
        
        sec.append(VerticalSpace('10pt'))

        alpha = np.round(mean).item()

        sum5_latex, res_sum5 = expand_sum("np.dot(({{}} - {})**2, {{}})".format(alpha), [DATA_VAR_FIRST[0, :], DATA_VAR_FIRST[1, :]])[1:3]

        res_sum5 = np.round(float(res_sum5), 4).item()

        sec.append(NoEscape(fr'$\alpha \approx \overline{{x}}_B = {alpha}$\\'))
        sec.append(NoEscape(fr'${format_math_str(Sum5._latex)} = {sum5_latex} = {res_sum5}$\\'))

        for i in range(len(gammas)):
            sec.append(VerticalSpace('10pt'))
            sec.append(NoEscape(fr'{i + 5}) $\gamma = {gammas[i]}\qquad \alpha = {alpha}$\\'))
            a, b = [np.round(i, 4).item() for i in (scipy.stats.chi2.ppf((1 + gammas[i]) / 2, count), scipy.stats.chi2.ppf((1 - gammas[i]) / 2, count))]
            range_ = [np.round(i, 4).item() for i in (res_sum5 / a, res_sum5 / b)]
            sec.append(NoEscape(fr'$\chi_{{\frac{{1+\gamma}}{{2}},n}}^2 = \chi_{{\frac{{1+{gammas[i]}}}{{2}},{count}}}^2 = {a}$\\'))
            sec.append(NoEscape(fr'$\chi_{{\frac{{1-\gamma}}{{2}},n}}^2 = \chi_{{\frac{{1-{gammas[i]}}}{{2}},{count}}}^2 = {b}$\\'))
            sec.append(VerticalSpace('5pt'))
            sec.append(NoEscape(r'$P\left(\frac{\sum_{i=1}^{n}{(x_i-a)^2}}{\chi_{\frac{1+\gamma}{2},n}^2} < \sigma^2 < \frac{\sum_{i=1}^{n}{(x_i-a)^2}}{\chi_{\frac{1-\gamma}{2},n}^2} \right) = ' + fr'P\left(\frac{{{res_sum5}}}{{{a}}} < \sigma^2 < \frac{{{res_sum5}}}{{{b}}}\right) = P\left({range_[0]} < \sigma^2 < {range_[1]} \right) = {gammas[i]}$\\'))

        sec.append(VerticalSpace('10pt'))

        S_squared = np.round(S**2, 4).item()

        sec.append(NoEscape(fr'$S^2 = {S}^2 = {S_squared}$\\'))

        for i in range(len(gammas)):
            sec.append(VerticalSpace('10pt'))
            sec.append(NoEscape(fr'{i + 7}) $\gamma = {gammas[i]}$\\'))
            a, b = [np.round(i, 4).item() for i in (scipy.stats.chi2.ppf((1 + gammas[i]) / 2, count-1), scipy.stats.chi2.ppf((1 - gammas[i]) / 2, count-1))]
            range_ = [np.round(i, 4).item() for i in (S_squared * (count - 1) / a, S_squared * (count - 1) / b)]
            eq = fr'{S}^2\cdot{count-1}'
            nl = (1-i) * r'\\'
            sec.append(NoEscape(fr'$\chi_{{\frac{{1+\gamma}}{{2}},n-1}}^2 = \chi_{{\frac{{1+{gammas[i]}}}{{2}},{count-1}}}^2 = {a}$\\'))
            sec.append(NoEscape(fr'$\chi_{{\frac{{1-\gamma}}{{2}},n-1}}^2 = \chi_{{\frac{{1-{gammas[i]}}}{{2}},{count-1}}}^2 = {b}$\\'))
            sec.append(VerticalSpace('5pt'))
            sec.append(NoEscape(r'$P\left(\frac{S^2(n-1)}{\chi_{\frac{1+\gamma}{2},n-1}^2} < \sigma^2 < \frac{S^2(n-1)}{\chi_{\frac{1-\gamma}{2},n-1}^2} \right) = ' + fr'P\left(\frac{{{eq}}}{{{a}}} < \sigma^2 < \frac{{{eq}}}{{{b}}}\right) = P\left({range_[0]} < \sigma^2 < {range_[1]} \right) = {gammas[i]}${nl}'))

        sec.append(LineBreak())
        sec.append(NewLine())

    intervals, n_i, w_i, p_norm, p_eq, F_star_n, F_norm_n, F_eq_n, F1_n, F2_n, F_eq_mp_n, F3_n = [None] * 12

    with doc.create(Section(HugeText('Работа №6'),numbering=False,label=False)) as sec:

        TABLE_TOTAL_1 = None

        intervals = [[INTERVALS[i].item(), INTERVALS[i+1].item()] for i in range(len(INTERVALS) - 1)]
        intervals_row = np.array([[fr'[{INTERVALS[i]},{INTERVALS[i+1]})' for i in range(len(INTERVALS) - 1)]])
        data_var = INTERVAL_VAR_NEW
        n_i = data_var[1, :].astype(int)
        n_i_row = n_i.astype(int).reshape((1, data_var.shape[1]))
        w_i = [frac(i.item(), count) for i in n_i]
        w_i_row = np.array([[str(frac(i.item(), count)) for i in n_i_row[0,:]]])

        p_norm_row = P_NORM.reshape((1,P_NORM.shape[0]))
        p_eq_row   = P_EQ.reshape((1, P_EQ.shape[0]))

        F_star_n = [sum(w_i[:i+1]) for i in range(len(w_i))]
        F_star_n_row = np.array([[str(i) for i in F_star_n]])

        F_norm_n = [np.round(np.sum(P_NORM[:i+1]), 4).item() for i in range(P_NORM.shape[0])]
        F_norm_n_row = np.array([F_norm_n])

        F_eq_n = [np.round(np.sum(P_EQ[:i+1]), 4).item() for i in range(P_EQ.shape[0])]
        F_eq_n_row = np.array([F_eq_n])

        F1_n = np.round(np.abs(np.array([i + 0.0 for i in F_star_n]) - F_norm_n), 4)
        F1_n_row = F1_n.reshape((1, F1_n.shape[0]))

        F2_n = np.round(np.abs(np.array([i + 0.0 for i in F_star_n]) - F_eq_n), 4)
        F2_n_row = F2_n.reshape((1, F2_n.shape[0]))



        a_hat = np.min(DATA_VAR_FIRST[0, :]).item()
        b_hat = np.max(DATA_VAR_FIRST[0, :]).item()

        p = np.zeros(data_var.shape[1])
        m = p.copy()

        sec.append(NoEscape(fr'$\hat{{\alpha}} = min\left(X_i\right) = {a_hat}\qquad \hat{{\beta}} = max(X_i) = {b_hat}$\\'))

        for i in range(len(p)):
            begin, end, h = calc_interval(INTERVALS[i:i+2], a_hat, b_hat)
            p[i] = np.round(1/(b_hat-a_hat) * h, 4)
            m[i] = np.round(count * p[i], 4)
            sec.append(NoEscape(fr'$p_{i+1} = \frac{{h}}{{\hat{{\beta}} - \hat{{\alpha}}}} = \frac{{{begin} - {end}}}{{{b_hat} - {a_hat}}} = {p[i].item()}\qquad m_{i+1} = n \cdot p_{i+1} = {count} \cdot {p[i]} = {m[i].item()}$\\'))

        sec.append(VerticalSpace('10pt'))

        chi_squared_eq_mmp_formula = r'\chi_{\text{РАВН(ММП)}}^2 = \sum_{i=1}^{k}{\left(\frac{(n_i-m_i)^2}{m_i}\right)}'
        chi_squared_eq_mmp_latex, res_chi_squared_eq_mpp = expand_sum("np.dot(({}-{})**2, 1/{})", [data_var[1, :], m, m])[1:3]

        res_chi_squared_eq_mpp = np.round(float(res_chi_squared_eq_mpp), 4).item()

        chi_squared_eq_mmp = res_chi_squared_eq_mpp

        sec.append(NoEscape(fr'${chi_squared_eq_mmp_formula} = {chi_squared_eq_mmp_latex} = {res_chi_squared_eq_mpp}$\\'))

        # Drawing table

        # F_norm_n = [np.round(np.sum(P_NORM[:i+1]), 4).item() for i in range(P_NORM.shape[0])]
        # F_norm_n_row = np.array([F_norm_n])

        F_eq_mp_n = np.array([np.round(np.sum(p[:i+1]), 4) for i in range(len(p))])
        F_eq_mp_n_row = F_eq_mp_n.reshape((1, F_eq_mp_n.shape[0]))

        F3_n = np.round(np.abs(np.array([i + 0.0 for i in F_star_n]) - F_eq_mp_n), 4)
        F3_n_row = F3_n.reshape((1, F3_n.shape[0]))

        TABLE_TOTAL_1 = np.concatenate((intervals_row, n_i_row, w_i_row, p_norm_row, p_eq_row, F_star_n_row, F_norm_n_row, F_eq_n_row, F_eq_mp_n_row, F1_n_row, F2_n_row, F3_n_row),axis=0)

        sec.append(VerticalSpace('5pt'))

        generate_random_var(sec, TABLE_TOTAL_1, [r'$I_i$', r'$n_i$', r'$w_i$', r'$p_{\text{норм}}$', r'$p_{\text{равн}}$', r'$F^*_n(x)$', r'$F_{\text{норм}}$', r'$F_{\text{равн}}$', r'$F_{\text{равн(МП)}}$', r'$|F^*_n(x)-F_{\text{норм}}(x)|$', r'$|F^*_n(x)-F_{\text{равн}}(x)|$', r'$|F^*_n(x) - F_{\text{равн(МП)}}(x)|$'])

        sec.append(LineBreak())
        sec.append(NewLine())

        # INTERVALS = np.array(x)
        # x = [[x[i], x[i+1]] for i in range(len(x)-1)]

        # interval_var = np.empty((2, len(x)), dtype=object)
        # interval_var[0, :] = [str(i).replace(']',')') for i in x]

    with doc.create(Section(HugeText('Работа №7'),numbering=False,label=False)) as sec:
        data_var = INTERVAL_VAR_NEW
        k = data_var.shape[1]
        r = 2
        df = k - r - 1

        alpha = 0.05
        # chi2_critical_value = np.round(scipy.stats.chi2.ppf(1 - alpha, df), 4).item()

        D_n_kolm = [np.max(F1_n), np.max(F2_n), np.max(F3_n)]
        ChiList = [chi_squared, chi_squared_eq, chi_squared_eq_mmp]

        with sec.create(Tabu('|c|c|c|c|c|',spread='15mm')) as table:
            table.add_hline()
            table.add_row((NoEscape(r'Распределение'), NoEscape(r'коэф. ас.'), NoEscape(r'коэф. экс.'), NoEscape(r'$\chi^2_{\text{Пир.}}$'), NoEscape(r'$D_{n_{\text{Колм.}}}(x)$')))
            table.add_hline()
            table.add_row((NoEscape(r'Нормал.'), a_b_u_3, e_b_u_4, ChiList[0], D_n_kolm[0]))
            table.add_hline(start=1, end=1)
            table.add_hline(start=4)
            table.add_row((NoEscape(r'а) Равномер. оц. пар. по Пир.'), '', '', ChiList[1], D_n_kolm[1]))
            table.add_hline(start=1, end=1)
            table.add_hline(start=4)
            table.add_row((NoEscape(r'б) Равномер. оц. пар. по ММП'), '', '', ChiList[2], D_n_kolm[2]))
            table.add_hline()

        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(fr'$k={k}\qquad r={r}\qquad df = k - r - 1 = {df}$\\'))

        range_end = int(max(ChiList)) + 2

        for i in range(len(D_n_kolm)):
            sec.append(VerticalSpace('10pt'))
            sec.append(NoEscape((r'1) Нормал.\\', r'2) Равномер. Пир.\\', r'3) Равномер. ММП.\\')[i]))
            sec.append(VerticalSpace('5pt'))
            sec.append(NoEscape(fr'''\begin{{tikzpicture}}[scale=0.5]
    \draw (0,0)-- ({range_end},0); %AXIS
    \foreach \x in {{0,{ChiList[i]}}} {'{'}
        \draw (\x,0.5) -- (\x,-0.5) node[below] {{$\x$}};
    {'}'}
    \draw (0,1) -- ({ChiList[i]}, 1);
    \draw[fill=white] (0,1) circle (0.25);
    \fill ({ChiList[i]}, 1) circle (0.25);
    \draw ({ChiList[i]}, 2) -- ({range_end},2);
    \fill ({ChiList[i]}, 2) circle (0.25);
    \draw[fill=white] ({range_end},2) circle (0.25);
    \end{{tikzpicture}}\\'''))
            
            alpha_a = np.round(1 - scipy.stats.chi2.cdf(ChiList[i], df=df), 4).item()
            alpha_b = np.round(scipy.stats.chi2.ppf(1 - alpha, df=df), 4).item()

            sec.append(VerticalSpace('10pt'))
            sec.append(NoEscape(fr'a) $\alpha=P\left( \chi^2 > {ChiList[i]} \right) = {alpha_a}'))

            if alpha_a > alpha:
                sec.append(NoEscape(fr' > {alpha} \implies $ не отвеграем\\'))
            else:
                sec.append(NoEscape(fr' \leq {alpha} \implies $ отвергаем\\'))
            
            sec.append(NoEscape(fr'б) $\alpha = {alpha} \qquad \chi_{{{alpha}}}^2({df}) = {alpha_b} \qquad'))

            if ChiList[i] < alpha_b:
                sec.append(NoEscape(fr'{ChiList[i]} < {alpha_b} \implies $ не отвергаем\\'))
            else:
                sec.append(NoEscape(fr'{ChiList[i]} \geq {alpha_b} \implies $ отвергаем\\'))
        
        sec.append(VerticalSpace('10pt'))
        sec.append(NoEscape(r'Критерий Колмогорова:\\'))

        for i in range(len(D_n_kolm)):
            lambda_0 = np.round(np.sqrt(count) * D_n_kolm[i], 4).item()
            prob = np.round(1 - scipy.stats.kstwobign.cdf(lambda_0), 4).item()
            sec.append(VerticalSpace('10pt'))
            sec.append(NoEscape((r'1) Нормал.\\', r'2) Равномер. Пир.\\', r'3) Равномер. ММП.\\')[i]))
            sec.append(NoEscape(fr'$\lambda_0 = \sqrt{{{count}}} \cdot {D_n_kolm[i]} = {lambda_0}$\\'))
            sec.append(NoEscape(fr'$P\left( \lambda_n > {lambda_0} \right) = {prob}$'))
            if i < len(D_n_kolm) - 1:
                sec.append(NoEscape(r'\\'))

        sec.append(LineBreak())
        sec.append(NewLine())
        sec.append(NewPage())

    with doc.create(Section(HugeText('Работа №8'),numbering=False,label=False)) as sec:
        m = GLOBAL_DATA['m1']
        n = GLOBAL_DATA['n1']

        A = np.array([m, m+n, m-4, m+n-1, m+2, m+n+1, m+n+4, 2*m, 2*n])
        B = np.array([n-1, n+m, n+4, 2*m-1, 2*n+1, m+3, m+n+2, m+n-3])
        C = np.array([m, n, 2*m+n, 2*n+m, m+n+3, m+n-2, 2*m+n-1, 2*n+m+3, 2*m+5, 2*n+4])

        n_s = [len(A), len(B), len(C)]

        probability = 0.05

        x_vals = [0] * 3
        s_vals = [0] * 3
        f_vals = []

        with sec.create(Tabu('|c|c|c|',spread='15mm')) as table:
            table.add_hline()
            table.add_row(('A', 'B', 'C'))
            table.add_hline()
            for i in range(np.max((A.size, B.size, C.size)).item()):
                table.add_row((A[i] if i < A.size else '', B[i] if i < B.size else '', C[i] if i < C.size else ''))
            table.add_hline()
            table.add_row((NoEscape(fr'$n_1={n_s[0]}$'), NoEscape(fr'$n_2={n_s[1]}$'), NoEscape(fr'$n_3={n_s[2]}$')))
            table.add_hline()

        sec.append(LineBreak())
        sec.append(NewLine())
        
        for i in range(3):
            arr = 0
            if i == 0: arr = A
            if i == 1: arr = B
            if i == 2: arr = C
            letter = ['A', 'B', 'C'][i]

            x_val_latex, x_val_res = expand_sum("np.sum({{}})/{}".format(n_s[i]), [arr])[1:3]
            x_val_res = np.round(float(x_val_res), 4).item()

            s_val_latex, s_val_res = expand_sum("np.sum(({{}} - {})**2)/{}".format(x_val_res, (n_s[i] - 1)), [arr])[1:3]
            s_val_res = np.round(float(s_val_res), 4).item()

            x_vals[i] = x_val_res
            s_vals[i] = s_val_res

            sec.append(NoEscape(fr'$\bar{{x}}_{{{letter}}} = \frac{{1}}{{n_{i+1}}}\cdot\sum_{{i=1}}^{{{len(arr)}}}{{{letter}_i}} = {x_val_latex} = \underline{{{x_val_res}}}$\\'))
            sec.append(VerticalSpace('5pt'))

            sec.append(NoEscape(fr'$S_{{{letter}}}^2 = \frac{{1}}{{n_{i+1}-1}}\cdot\sum_{{i=1}}^{{{len(arr)}}}{{({letter}_i - \bar{{x}}_{{{letter}}})^2}} = {s_val_latex} = \underline{{{s_val_res}}}$\\'))
            sec.append(VerticalSpace('5pt'))
        
        sec.append(VerticalSpace('5pt'))

        for i, j in ((0, 1), (0, 2), (1, 2)):

            # arr1 = 0
            # if i == 0: arr1 = A
            # if i == 1: arr1 = B
            # if i == 2: arr1 = C
            letter1 = ['A', 'B', 'C'][i]

            # arr2 = 0
            # if j == 0: arr2 = A
            # if j == 1: arr2 = B
            # if j == 2: arr2 = C
            letter2 = ['A', 'B', 'C'][j]

            f = np.round(max(s_vals[i], s_vals[j]) / min(s_vals[i], s_vals[j]), 4).item()
            f_vals.append(f)
            n_arr = sorted((n_s[i]-1,n_s[j]-1), reverse=True)
            df1, df2 = n_arr
            f_p_val = np.round(scipy.stats.f.ppf(1 - probability, df1, df2), 4).item()

            n_str = str(n_arr).replace('[','(').replace(']',')')

            d = [str('<' if f < f_p_val else '>'), str('принимаем' if f < f_p_val else 'отвергаем')]

            sec.append(NoEscape(fr'$D_{letter1} , D_{letter2}:\qquad F = \frac{{{max(s_vals[i], s_vals[j])}}}{{{min(s_vals[i], s_vals[j])}}} = {f} \qquad \underset{{{n_str}}}{{F_{{\text{{табл.}}}}}} = {f_p_val}$\\'))
            sec.append(NoEscape(fr'$F {d[0]} F_{{\text{{табл.}}}} \implies $ не {d[1]} при $\alpha = {probability}$\\'))
            sec.append(VerticalSpace('5pt'))

        sec.append(VerticalSpace('5pt'))

        for i, j in ((0, 1), (0, 2), (1, 2)):
            letter1 = ['A', 'B', 'C'][i]
            letter2 = ['A', 'B', 'C'][j]

            t = np.round((np.abs(x_vals[i] - x_vals[j]) / np.sqrt(n_s[i] * s_vals[i] + n_s[j] * s_vals[j])) * np.sqrt((n_s[i] * n_s[j] * (n_s[i] + n_s[j] - 2)) / (n_s[i] + n_s[j])), 4).item()
            t_crit = np.round(scipy.stats.t.ppf(1 - probability / 2, n_s[i] + n_s[j] - 2), 4).item()

            t_latex = fr't_{{{letter1}, {letter2}}} = \frac{{{max(x_vals[i], x_vals[j])} - {min(x_vals[i], x_vals[j])}}}{{\sqrt{{{n_s[i]}\cdot {s_vals[i]} + {n_s[j]}\cdot {s_vals[j]}}}}}\cdot \sqrt{{\frac{{{n_s[i]}\cdot {n_s[j]}\cdot \left({n_s[i]} + {n_s[j]} - 2 \right)}}{{{n_s[i]} + {n_s[j]}}}}}'
            t_crit_str = fr'({n_s[i] + n_s[j] - 2}, {probability})'

            d = [str('<' if abs(t) < t_crit else '>'), str('принимаем' if abs(t) < t_crit else 'отвергаем')]

            sec.append(NoEscape(fr'${t_latex} = \underline{{{t}}}\qquad \underset{{{t_crit_str}}}{{t_{{\text{{табл.}}}}}} = {t_crit}$\\'))
            sec.append(NoEscape(fr'$\left| t \right| {d[0]} t_{{\text{{табл.}}}} \implies MX = MY $ {d[1]} на уровне значения $\alpha = {probability}$\\'))
            sec.append(VerticalSpace('5pt'))
            
            sec.append(VerticalSpace('5pt'))
        
        ARRAYS = [A, B, C]
        F_VALS = f_vals
        X_VALS = x_vals
        S_VALS = s_vals

            #    chi_squared_eq_mmp_latex, res_chi_squared_eq_mpp = expand_sum("np.dot(({}-{})**2, 1/{})", [data_var[1, :], m, m])[1:3]
            #    res_chi_squared_eq_mpp = np.round(float(res_chi_squared_eq_mpp), 4).item()
        sec.append(NewPage())

    with doc.create(Section(HugeText('Работа №9'),numbering=False,label=False)) as sec:
        A, B, C = [i.copy() for i in ARRAYS]
        x_vals = X_VALS
        s_vals = S_VALS
        f_vals = F_VALS

        n_s = [len(i) for i in ARRAYS]

        i1, i2 = ((0, 1), (0, 2), (1, 2))[f_vals.index(max(f_vals))]

        letter1 = ['A', 'B', 'C'][i1]
        letter2 = ['A', 'B', 'C'][i2]

        array1 = ARRAYS[i1]
        array2 = ARRAYS[i2]

        probability = 0.05

        with sec.create(Tabu('|c|c|c|c|',spread='15mm')) as table:
            table.add_hline()
            table.add_row(('', 'A', 'B', 'C'))
            table.add_hline()
            table.add_row([NoEscape(r'$\overline{x}$')] + [NoEscape(fr'${i}$') for i in x_vals])
            table.add_hline()
            table.add_row([NoEscape(r'$S^2$')] + [str(i) for i in s_vals])
            table.add_hline()
            table.add_row([NoEscape(r'$n_i$')] + [str(len(i)) for i in ARRAYS])
            table.add_hline()
        
        sec.append(LineBreak())
        sec.append(NewLine())

        sec.append(NoEscape(fr'Для ${letter1}, {letter2}$:\\'))

        t_latex, t_res = expand_expr_new(fr'({x_vals[i1]} - {x_vals[i2]}) / sqrt({s_vals[i1]} / {n_s[i1]} + {s_vals[i2]} / {n_s[i2]})')
        
        # t_latex, t_res = expand_expr("({} - {}) / sp.sqrt({} / {} + {} / {})", [x_vals[i1], x_vals[i2], s_vals[i1], n_s[i1], s_vals[i2], n_s[i2]])[1:3]
        
        df_latex, df_res = expand_expr_new(fr'(({s_vals[i1]} / {n_s[i1]} + {s_vals[i2]} / {n_s[i2]}) ** 2) / ((({s_vals[i1]} / {n_s[i1]})**2)/({n_s[i1]}+1) + (({s_vals[i2]} / {n_s[i2]})**2)/({n_s[i2]}+1)) - 2')

        val1 = np.floor(df_res).item()
        val2 = val1 + 1

        crit_t_val1 = np.round(scipy.stats.t.ppf(1-probability/2,df=val1), 4).item()
        crit_t_val2 = np.round(scipy.stats.t.ppf(1-probability/2,df=val2), 4).item()
        crit_t_val3 = np.round(scipy.stats.t.ppf(1-probability/2,df=df_res), 4).item()

        x, y = sp.symbols('x y')
        equation = sp.Eq((y - crit_t_val1)/(crit_t_val2 - crit_t_val1), (x - val1)/(val2 - val1))
        solution = round_expr(sp.solve(equation, y)[0], 4)
        # x_function = sp.lambdify(y, solution)

        arr = np.array([crit_t_val1, crit_t_val2])
        idx = (np.abs(arr - crit_t_val3)).argmin()
        crit_t_val3 = np.round(arr[idx], 4).item()
        


        sec.append(NoEscape(fr'$t = {t_latex} = {t_res}$\\'))
        sec.append(NoEscape(fr'$df = {df_latex} = {df_res}$\\'))

        sec.append(VerticalSpace('10pt'))

        sec.append(NoEscape(fr'''$\begin{{array}}{{rcl}}
    T_{{{probability}}}({val1}) & = & {crit_t_val1}  \\
    T_{{{probability}}}({val2}) & = & {crit_t_val2}
    \end{{array}} \implies '''))
        
        sec.append(NoEscape(fr'\frac{{y-{crit_t_val1}}}{{{crit_t_val2}-{crit_t_val1}}}=\frac{{x-{val1}}}{{{val2}-{val1}}} \implies'))
        sec.append(NoEscape(fr'y = {sp.latex(solution)} \implies$\\'))
        with sec.create(FlushRight()) as env:
            env.append(NoEscape(fr'$\implies T_{{{probability}}}({df_res}) \approx {crit_t_val3}$\\'))
        sec.append(VerticalSpace('5pt'))

        b = bool(abs(t_res) < crit_t_val3)
        d = ['<' if b else '>', 'не' if b else '', '=' if b else r'\neq']
        sec.append(NoEscape(fr'$\left|{t_res} \right| {d[0]} {crit_t_val3} \implies H {d[1]} $ принимаем, $M{letter1} {d[2]} M{letter2}$\\'))

        sec.append(VerticalSpace('10pt'))

        # B, A = np.array_split(np.concat((A, B, C)), 2)

        letter1, letter2 = 'A', 'B'

        A = np.concat((A, B[:B.size//2]))
        B = np.concat((B[B.size//2:],C))

        sec.append(NoEscape(fr'$A: {A.tolist()} \qquad n_{letter1} = {len(A)}$\\'))
        sec.append(NoEscape(fr'$B: {B.tolist()} \qquad n_{letter2} = {len(B)}$\\'))

        x_a_latex, x_a_res = expand_sum(f"np.sum({{}})/{len(A)}", [A])[1:3]
        x_b_latex, x_b_res = expand_sum(f"np.sum({{}})/{len(B)}", [B])[1:3]

        d_a_latex, d_a_res = expand_sum(f"np.sum(({{}} - {x_a_res})**2)/{len(A)}", [A])[1:3]
        d_b_latex, d_b_res = expand_sum(f"np.sum(({{}} - {x_b_res})**2)/{len(B)}", [B])[1:3]

        sec.append(NoEscape(fr'$\overline{{x}}_{letter1} = {x_a_latex} = \underline{{{x_a_res}}}$\\'))
        sec.append(NoEscape(fr'$\overline{{x}}_{letter2} = {x_b_latex} = \underline{{{x_b_res}}}$\\'))

        sec.append(NoEscape(fr'$D_{letter1}(x) = {d_a_latex} = \underline{{{d_a_res}}}$\\'))
        sec.append(NoEscape(fr'$D_{letter2}(x) = {d_b_latex} = \underline{{{d_b_res}}}$\\'))
        sec.append(VerticalSpace('10pt'))

        z_latex, z_res = expand_expr_new(fr'({x_b_res} - {x_a_res}) / sqrt({d_a_res} / {len(A)} + {d_b_res} / {len(B)})')

        gamma = 1 - probability

        z_crit = np.round(scipy.stats.t.ppf(1 - 0.3 / 2, 1), 4).item()

        sec.append(NoEscape(fr'$z = {z_latex} = {z_res}$\\'))
        sec.append(VerticalSpace('10pt'))

        sec.append(NoEscape(fr'$\gamma = {gamma} \qquad z_{{\text{{кр.}}}} = {z_crit}$\\'))

        b = bool(abs(z_res) < z_crit)
        d = ['<' if b else '>', 'не' if b else '', '=' if b else r'\neq']
        sec.append(NoEscape(fr'$\left| {z_res} \right| {d[0]} {z_crit} \implies $ {d[1]} принимаем, $M{letter1} {d[2]} M{letter2}$\\'))

    filename = r'{}-{}'.format(GLOBAL_DATA['ФИО'], GLOBAL_DATA["Группа"])
    if (bTex and not bPdf):
        doc.generate_tex(filename)
    elif (bTex and bPdf):
        doc.generate_pdf(filename,clean=True, clean_tex=False)
    elif (bPdf and not bTex):
        doc.generate_pdf(filename,clean=True, clean_tex=True)
    # doc.generate_pdf(filepath='statisticspdf',clean=True,clean_tex=False)



if __name__ == "__main__":
    main()
