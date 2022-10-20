from __future__ import print_function

import argparse
import functools
import re
import warnings
import sys
import collections
import pandas as pd
from IPython import embed
import yaml
import ipdb
import numpy as np

from grg_pssedata.struct import Bus
from grg_pssedata.struct import Load
from grg_pssedata.struct import FixedShunt
from grg_pssedata.struct import Generator
from grg_pssedata.struct import Branch
from grg_pssedata.struct import TwoWindingTransformer
from grg_pssedata.struct import ThreeWindingTransformer
from grg_pssedata.struct import TransformerParametersFirstLine
from grg_pssedata.struct import TransformerParametersSecondLine
from grg_pssedata.struct import TransformerParametersSecondLineShort
from grg_pssedata.struct import TransformerWinding
from grg_pssedata.struct import TransformerWindingShort
from grg_pssedata.struct import Area
from grg_pssedata.struct import Zone
from grg_pssedata.struct import Owner
from grg_pssedata.struct import SwitchedShunt
from grg_pssedata.struct import Case
from grg_pssedata.struct import TwoTerminalDCLine
from grg_pssedata.struct import TwoTerminalDCLineParameters
from grg_pssedata.struct import TwoTerminalDCLineRectifier
from grg_pssedata.struct import TwoTerminalDCLineInverter
from grg_pssedata.struct import VSCDCLine
from grg_pssedata.struct import VSCDCLineParameters
from grg_pssedata.struct import VSCDCLineConverter
from grg_pssedata.struct import TransformerImpedanceCorrection
from grg_pssedata.struct import MultiTerminalDCLine
from grg_pssedata.struct import MultiTerminalDCLineParameters
from grg_pssedata.struct import MultiTerminalDCLineConverter
from grg_pssedata.struct import MultiTerminalDCLineDCBus
from grg_pssedata.struct import MultiTerminalDCLineDCLink
from grg_pssedata.struct import MultiSectionLineGrouping
from grg_pssedata.struct import InterareaTransfer
from grg_pssedata.struct import FACTSDevice
from grg_pssedata.struct import InductionMachine

from grg_pssedata.exception import PSSEDataParsingError
from grg_pssedata.exception import PSSEDataWarning

with open('headers.yaml', 'r') as f:
    try:
        HEADERS = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

with open('areas.yaml', 'r') as f:
    try:
        AREAS = yaml.safe_load(f)
        INTERNALS = AREAS['ISONE'] + AREAS['NYISO'] + AREAS['PJM'] + AREAS['DUKE'] + AREAS['SC']
        EXTERNALS = list(set(AREAS['ALLAREAS']) - set(INTERNALS))
        # INTERNALS = list(set(AREAS['ALLAREAS']) - set(EXTERNALS))
        # EXTERNALS = AREAS['ISONE'] + AREAS['NYISO'] + AREAS['PJM'] + AREAS['DUKE'] + AREAS['SC']
    except yaml.YAMLError as exc:
        print(exc)


CAPLIM = 500 # 50 MW limit of substations
KVLIM = 345 # 345 kV lines and above
CONLIM = 3 # number of connections to a bus

LineRequirements = collections.namedtuple('LineRequirements',['line_index','min_values','max_values','section'])

print_err = functools.partial(print, file=sys.stderr)

psse_table_terminus = '0'
psse_record_terminus = 'Q'
psse_terminuses = [psse_table_terminus, psse_record_terminus]

def expand_commas(list):
    expanded_list = []
    for item in list:
        if not ',' in item:
            expanded_list.append(item)
        else:
            for i in range(0, item.count(',')-1):
                expanded_list.append(None)
    return expanded_list


def parse_psse_case_file(psse_file_name):
    '''opens the given path and parses it as pss/e data

    Args:
        psse_file_name(str): path to the a psse data file
    Returns:
        Case: a grg_pssedata case
    '''

    with open(psse_file_name, 'r') as psse_file:
        lines = psse_file.readlines()

    #try:
    psse_data = parse_psse_case_lines(lines)
    #except BaseException as e:
    #    raise PSSEDataParsingError('{}'.format(str(e)))

    return psse_data


def parse_psse_case_str(psse_string):
    '''parses a given string as matpower data

    Args:
        mpString(str): a matpower data file as a string
    Returns:
        Case: a grg_pssedata case
    '''

    lines = psse_string.split('\n')

    #try:
    psse_data = parse_psse_case_lines(lines)
    #except BaseException as e:
    #    raise PSSEDataParsingError('{}'.format(str(e)))

    return psse_data



def parse_line(line, line_reqs=None):
    line = line.strip()
    comment = None

    l = re.split(r"(?!\B[\"\'][^\"\']*)[\/](?![^\"\']*[\"\']\B)", line, maxsplit=1)
    if len(l) > 1:
        line, comment = l
    else:
        line = l[0]

    line_parts = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", line)

    if line_reqs is not None:
        if len(line_parts) < line_reqs.min_values:
            raise PSSEDataParsingError('on psse data line {} in the "{}" section, at least {} values were expected but only {} where found.\nparsed: {}'.format(line_reqs.line_index, line_reqs.section, line_reqs.min_values, len(line_parts), line_parts))
        if len(line_parts) > line_reqs.max_values:
            warnings.warn('on psse data line {} in the "{}" section, at most {} values were expected but {} where found, extra values will be ignored.\nparsed: {}'.format(line_reqs.line_index, line_reqs.section, line_reqs.max_values, len(line_parts), line_parts), PSSEDataWarning)
            line_parts = line_parts[:line_reqs.max_values]

    return line_parts, comment


def parse_psse_case_lines(lines):
    if len(lines) < 3: # need at base values and record
        raise PSSEDataParsingError('psse case has {} lines and at least 3 are required'.format(len(lines)))

    (ic, sbase, rev, xfrrat, nxfrat, basefrq), comment = parse_line(lines[0], LineRequirements(0, 6, 6, "header"))
    print_err('case data: {} {} {} {} {} {}'.format(ic, sbase, rev, xfrrat, nxfrat, basefrq))

    if len(ic.strip()) > 0 and not (ic.strip() == "0"): # note validity checks may fail on "change data"
        raise PSSEDataParsingError('ic value of {} given, only a value of 0 is supported'.format(ic))

    version_id = 33
    if len(rev.strip()) > 0:
        try:
            version_id = int(float(rev))
        except ValueError:
             warnings.warn('assuming PSSE version 33, given version value "{}".'.format(rev.strip()), PSSEDataWarning)

    if version_id != 33:
        warnings.warn('PSSE version {} given but only version 33 is supported, parser may not function correctly.'.format(rev.strip()), PSSEDataWarning)

    record1 = lines[1].strip('\n')
    record2 = lines[2].strip('\n')
    print_err('record 1: {}'.format(record1))
    print_err('record 2: {}'.format(record2))

    buses = []
    loads = []
    fixed_shunts = []
    generators = []
    branches = []
    transformers = []
    transformers3w = []
    transformers2w = []
    areas = []
    tt_dc_lines = []
    vsc_dc_lines = []
    transformer_corrections = []
    mt_dc_lines = []
    line_groupings = []
    zones = []
    transfers = []
    owners = []
    facts = []
    switched_shunts = []
    gnes = []
    induction_machines = []


    line_index = 3
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 9, 13, "bus"))
        buses.append(Bus(*line_parts).__df__())
        line_index += 1
    print_err('parsed {} buses'.format(len(buses)))
    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    Busdf = pd.DataFrame(data=buses, columns=HEADERS['bus'])

    load_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 13, 14, "load"))
        loads.append(Load(line_index - load_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} loads'.format(len(loads)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    Loaddf = pd.DataFrame(data=loads, columns=HEADERS['load'])

    fixed_shunt_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 5, 5, "fixed shunt"))
        fixed_shunts.append(FixedShunt(line_index - fixed_shunt_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} fixed shunts'.format(len(fixed_shunts)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    fixshuntdf = pd.DataFrame(data=fixed_shunts, columns=HEADERS['fixshunt'])

    gen_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 20, 28, "generator"))
        generators.append(Generator(line_index - gen_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} generators'.format(len(generators)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    gensdf = pd.DataFrame(data=generators, columns=HEADERS['generator'])

    branch_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        #line = shlex.split(lines[line_index].strip())
        #line = expand_commas(line)
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 18, 24, "branch"))
        #print(line_parts)
        branches.append(Branch(line_index - branch_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} branches'.format(len(branches)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    branchesdf = pd.DataFrame(data=branches, columns=HEADERS['acline'])

    transformer_index = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts_1, comment_1 = parse_line(lines[line_index], LineRequirements(line_index, 20, 21, "transformer"))
        parameters_1 = TransformerParametersFirstLine(*line_parts_1)
        #print(parameters_1)

        if parameters_1.k == 0: # two winding case
            line_parts_2, comment_2 = parse_line(lines[line_index+1], LineRequirements(line_index+1, 3, 3, "transformer"))
            line_parts_3, comment_3 = parse_line(lines[line_index+2], LineRequirements(line_index+1, 16, 17, "transformer"))
            line_parts_4, comment_4 = parse_line(lines[line_index+3], LineRequirements(line_index+1, 2, 2, "transformer"))

            parameters_2 = TransformerParametersSecondLineShort(*line_parts_2)
            winding_1 = TransformerWinding(1, *line_parts_3)
            winding_2 = TransformerWindingShort(2, *line_parts_4)

            t = TwoWindingTransformer(transformer_index, parameters_1.__df__(), parameters_2.__df__(), winding_1.__df__(), winding_2.__df__()).__df__()
            transformers2w.append(sum(t,[]))

            line_index += 4
        else: # three winding case
            line_parts_2, comment_2 = parse_line(lines[line_index+1], LineRequirements(line_index+1, 11, 11, "transformer"))
            line_parts_3, comment_3 = parse_line(lines[line_index+2], LineRequirements(line_index+2, 17, 17, "transformer"))
            line_parts_4, comment_4 = parse_line(lines[line_index+3], LineRequirements(line_index+3, 17, 17, "transformer"))
            line_parts_5, comment_5 = parse_line(lines[line_index+4], LineRequirements(line_index+4, 17, 17, "transformer"))

            parameters_2 = TransformerParametersSecondLine(*line_parts_2)
            winding_1 = TransformerWinding(1, *line_parts_3)
            winding_2 = TransformerWinding(2, *line_parts_4)
            winding_3 = TransformerWinding(3, *line_parts_5)

            t = ThreeWindingTransformer(transformer_index, parameters_1.__df__(), parameters_2.__df__(), winding_1.__df__(), winding_2.__df__(), winding_3.__df__()).__df__()
            transformers3w.append(sum(t,[]))

            line_index += 5

        transformers.append(sum(t,[]))
        transformer_index += 1

    print_err('parsed {} transformers'.format(len(transformers)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    trans3wdf = pd.DataFrame(data=transformers3w, columns=HEADERS['transformer3w'])
    trans2wdf = pd.DataFrame(data=transformers2w, columns=HEADERS['transformer2w'])

    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 1, 5, "areas"))
        areas.append(Area(*line_parts).__df__())
        line_index += 1
    print_err('parsed {} areas'.format(len(areas)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    areasdf = pd.DataFrame(data=areas, columns=HEADERS['area'])

    #two terminal dc line data
    ttdc_index = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts_1, comment_1 = parse_line(lines[line_index], LineRequirements(line_index, 12, 12, "two terminal dc line"))
        line_parts_2, comment_2 = parse_line(lines[line_index+1], LineRequirements(line_index+1, 17, 17, "two terminal dc line"))
        line_parts_3, comment_3 = parse_line(lines[line_index+2], LineRequirements(line_index+2, 17, 17, "two terminal dc line"))

        parameters = TwoTerminalDCLineParameters(*line_parts_1)
        rectifier = TwoTerminalDCLineRectifier(*line_parts_2)
        inverter = TwoTerminalDCLineInverter(*line_parts_3)

        tt_dc_lines.append(sum(TwoTerminalDCLine(ttdc_index, parameters.__df__(), rectifier.__df__(), inverter.__df__()).__df__(),[]))

        ttdc_index += 1
        line_index += 3
    print_err('parsed {} two terminal dc lines'.format(len(tt_dc_lines)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    tt_dc_linesdf = pd.DataFrame(data=tt_dc_lines, columns=HEADERS['twotermdc'])

    #vsc dc line data
    vscdc_index = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts_1, comment_1 = parse_line(lines[line_index], LineRequirements(line_index, 3, 11, "vsc dc line"))
        line_parts_2, comment_2 = parse_line(lines[line_index+1], LineRequirements(line_index+1, 13, 15, "vsc dc line"))
        line_parts_3, comment_3 = parse_line(lines[line_index+2], LineRequirements(line_index+2, 13, 15, "vsc dc line"))

        parameters = VSCDCLineParameters(*line_parts_1)
        converter_1 = VSCDCLineConverter(*line_parts_2)
        converter_2 = VSCDCLineConverter(*line_parts_3)

        vsc_dc_lines.append(sum(VSCDCLine(vscdc_index, parameters.__df__(), converter_1.__df__(), converter_2.__df__()).__df__(),[]))

        line_index += 3
        vscdc_index += 1
    print_err('parsed {} vsc dc lines'.format(len(vsc_dc_lines)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    vsc_dc_linesdf = pd.DataFrame(data=vsc_dc_lines, columns=HEADERS['vscdc'])

    #transformer impedence correction tables data
    trans_offset_index = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 1, 23, "transformer correction"))
        transformer_corrections.append(TransformerImpedanceCorrection(line_index - trans_offset_index, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} transformer corrections'.format(len(transformer_corrections)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    # XXX
    # trans_cor_df = pd.DataFrame(data=transformer_corrections, columns=HEADERS['vscdc'])

    #multi-terminal dc line data
    mtdc_count = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 8, 8, "multi-terminal dc line"))
        parameters = MultiTerminalDCLineParameters(*line_parts)

        nconv, ndcbs, ndcln = [], [], []
        for i in range(0, parameters.nconv):
            line_parts, comment = parse_line(lines[line_index + i + 1], LineRequirements(line_index + i + 1, 16, 16, "multi-terminal dc line"))
            nconv.append(MultiTerminalDCLineConverter(*line_parts).__df__())

        for i in range(parameters.nconv, parameters.ndcbs+parameters.nconv):
            line_parts, comment = parse_line(lines[line_index + i + 1], LineRequirements(line_index + i + 1, 8, 8, "multi-terminal dc line"))
            ndcbs.append(MultiTerminalDCLineDCBus(*line_parts).__df__())

        for i in range(parameters.nconv + parameters.ndcbs, parameters.ndcln+parameters.nconv+parameters.ndcbs):
            line_parts, comment = parse_line(lines[line_index + i + 1], LineRequirements(line_index + i + 1, 6, 6, "multi-terminal dc line"))
            ndcln.append(MultiTerminalDCLineDCLink(*line_parts).__df__())

        mt_dc_lines.append(MultiTerminalDCLine(mtdc_count, parameters, nconv, ndcbs, ndcln).__df__())
        mtdc_count += 1
        line_index += 1 + parameters.nconv + parameters.ndcbs + parameters.ndcln
    print_err('parsed {} multi-terminal dc lines'.format(len(mt_dc_lines)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    #multi-section line grouping data
    print('parsing multisection lines')
    msline_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 5, 5, "multi-section line"))
        line_groupings.append(MultiSectionLineGrouping(line_index - msline_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} multi-section lines'.format(len(line_groupings)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1


    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 2, 2, "zone"))
        zones.append(Zone(*line_parts).__df__())
        line_index += 1
    print_err('parsed {} zones'.format(len(zones)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    zonesdf = pd.DataFrame(data=zones, columns=HEADERS['zone'])

    # inter area transfer data
    intarea_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 4, 4, "inter-area transfer"))
        transfers.append(InterareaTransfer(line_index - intarea_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} inter-area transfers'.format(len(transfers)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 2, 2, "owner"))
        owners.append(Owner(*line_parts).__df__())
        line_index += 1
    print_err('parsed {} owners'.format(len(owners)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    ownersdf = pd.DataFrame(data=owners, columns=HEADERS['owner'])

    # facts device data block
    facts_index = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 19, 21, "facts device"))
        facts.append(FACTSDevice(facts_index, *line_parts).__df__())
        facts_index += 1
        line_index += 1
    print_err('parsed {} facts devices'.format(len(facts)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    factsdf = pd.DataFrame(data=facts, columns=HEADERS['facts'])

    # switched shunt data block
    swithced_shunt_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 12, 26, "swticthed shunt"))
        switched_shunts.append(SwitchedShunt(line_index - swithced_shunt_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} switched shunts'.format(len(switched_shunts)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    swshuntdf = pd.DataFrame(data=switched_shunts, columns=HEADERS['swshunt'])

    # GNE device data
    gne_count = 0
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        gne_count += 1
        line_index += 1
    if gne_count > 0:
        warnings.warn('skipped {} lines of GNE data'.format(gne_count), PSSEDataWarning)
        #print_err('parsed {} generic network elements'.format(len(gnes)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    # induction machine data
    indm_index_offset = line_index
    while parse_line(lines[line_index])[0][0].strip() not in psse_terminuses:
        line_parts, comment = parse_line(lines[line_index], LineRequirements(line_index, 34, 34, "induction machine"))
        induction_machines.append(InductionMachine(line_index - indm_index_offset, *line_parts).__df__())
        line_index += 1
    print_err('parsed {} induction machines'.format(len(induction_machines)))

    if parse_line(lines[line_index])[0][0].strip() != psse_record_terminus:
        line_index += 1

    print_err('un-parsed lines:')
    while line_index < len(lines):
        #print(parse_line(lines[line_index]))
        print_err('  '+lines[line_index])
        line_index += 1

    return Busdf, Loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf


def read_psse(args):
    Busdf, Loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf = parse_psse_case_file(args.file)
    return Busdf, Loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='the pss/e data file to operate on (.raw)')

    return parser


# if __name__ == '__main__':
#     parser = build_cli_parser()
#     Busdf, Loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf = read_psse(parser.parse_args())

def fix_trans3w(trans3wpre):
    trans3wpre.loc[trans3wpre['stat']==1,'nl']=3
    trans3wpre.loc[trans3wpre['stat']==0,'nl']=0
    trans3wpre.loc[trans3wpre['stat'].isin([2,3,4]),'nl']=2
    print(f"total number of lines {trans3wpre['nl'].sum()}")
    trans3wpre = trans3wpre.drop(columns='nl')
    t3w=[]
    for ibus, jbus, kbus, ckt, cw, cz, name, stat,  vecgrp, r1_2, x1_2, sbase1_2, r2_3, x2_3, sbase2_3, r3_1, x3_1, sbase3_1, wdg1rate1, wdg1rate2, wdg1rate3, wdg2rate1, wdg2rate2, wdg2rate3, wdg3rate1, wdg3rate2, wdg3rate3 in trans3wpre.values:
        if stat ==1:
            t3w.append([ibus, jbus, ckt, cw, cz, name, stat,  vecgrp, r1_2, x1_2, sbase1_2, wdg1rate1, wdg1rate2, wdg1rate3])
            t3w.append([jbus, kbus, ckt, cw, cz, name, stat,  vecgrp, r2_3, x2_3, sbase2_3, wdg2rate1, wdg2rate2, wdg2rate3])
            t3w.append([ibus, kbus, ckt, cw, cz, name, stat,  vecgrp, r3_1, x3_1, sbase3_1, wdg3rate1, wdg3rate2, wdg3rate3])
        elif stat==2: # only winding 2 is out
            t3w.append([ibus, jbus, ckt, cw, cz, name, stat,  vecgrp, r1_2, x1_2, sbase1_2, wdg1rate1, wdg1rate2, wdg1rate3])
            t3w.append([ibus, kbus, ckt, cw, cz, name, stat,  vecgrp, r3_1, x3_1, sbase3_1, wdg3rate1, wdg3rate2, wdg3rate3])
        elif stat==3: # only winding 3 is out: add 1 and 2
            t3w.append([ibus, jbus, ckt, cw, cz, name, stat,  vecgrp, r1_2, x1_2, sbase1_2, wdg1rate1, wdg1rate2, wdg1rate3])
            t3w.append([jbus, kbus, ckt, cw, cz, name, stat,  vecgrp, r2_3, x2_3, sbase2_3, wdg2rate1, wdg2rate2, wdg2rate3])
        elif stat==1: # only winding 1 is out: add 2 and 3
            t3w.append([jbus, kbus, ckt, cw, cz, name, stat,  vecgrp, r2_3, x2_3, sbase2_3, wdg2rate1, wdg2rate2, wdg2rate3])
            t3w.append([ibus, kbus, ckt, cw, cz, name, stat,  vecgrp, r3_1, x3_1, sbase3_1, wdg3rate1, wdg3rate2, wdg3rate3])
    return pd.DataFrame(data=t3w, columns=['ibus', 'jbus', 'ckt', 'cw', 'cz', 'name', 'stat', 'vecgrp', 'r', 'x', 'sbase', 'rate1', 'rate2', 'rate3'])

def fix_tt_dc(df):
    '''calculate p, qmin, and qmin for rectifier and inverter
    '''
    # XXX the Qmin becomes larger than Qmax. This is the same as matpower; WHY? XXX

    df.loc[df['mdc']==1,'pmw']=df.loc[df['mdc']==1,'setvl'] # SETVL is the desired real power demand
    df.loc[df['mdc']==2,'pmw']=df.loc[df['mdc']==2,'setvl'] * df.loc[df['mdc']==2,'vschd'] / 1000  # SETVL is the current in amps (need devide 1000 to convert to MW)
    df.loc[~(df['mdc'].isin([1,2])),'pmw']=0
    df.loc[df['pmw']<0, 'pmw'] = -1 * df.loc[df['pmw']<0, 'pmw']

    # Q min and max for rectifires and inverters
    df.loc[:, 'qminr'] = df['pmw'] * df['anmnr']
    df.loc[:, 'qmini'] = df['pmw'] * df['anmni']

    df.loc[:, 'qmaxr'] = df['pmw'] * np.tan(np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxr'])) + np.cos(np.pi/3))))
    df.loc[:, 'qmaxi'] = df['pmw'] * np.tan(np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxi'])) + np.cos(np.pi/3))))
    df.loc[(df['qminr']<0), 'qminr'] = -1 * df.loc[(df['qminr']<0), 'qminr']
    df.loc[(df['qmini']<0), 'qmini'] = -1 * df.loc[(df['qmini']<0), 'qmini']
    df.loc[(df['qmaxr']<0), 'qmaxr'] = -1 * df.loc[(df['qmaxr']<0), 'qmaxr']
    df.loc[(df['qmaxi']<0), 'qmaxi'] = -1 * df.loc[(df['qmaxi']<0), 'qmaxi']
    df.loc[df['mdc']==0, 'stat'] = 0
    df.loc[df['mdc']!=0, 'stat'] = 1

    return df


def calc_buscap(allbranches, buson):

    # ########################################################################################### #
    # calculate capacity of each substation
    # ########################################################################################### #

    # for ibus
    subcap = pd.DataFrame()
    ic = allbranches.groupby(['ibus'])['jbus'].count().reindex(buson['ibus']).fillna(value=0)
    jc = allbranches.groupby(['jbus'])['ibus'].count().reindex(buson['ibus']).fillna(value=0)
    ijc = ic + jc
    ijc.name = 'count'
    ijc = ijc.reset_index()

    ibus = allbranches.groupby(['ibus'])[['rate2']].agg({'rate2':['sum', 'max']}).reindex(buson['ibus']).fillna(value=0)['rate2']
    jbus = allbranches.groupby(['jbus'])[['rate2']].agg({'rate2':['sum', 'max']}).reindex(buson['ibus']).fillna(value=0)['rate2']

    subcap['sum'] = ibus['sum'] + jbus['sum']
    subcap['max'] = np.maximum(ibus['max'], jbus['max'])
    subcap['cap'] = subcap['sum'] - subcap['max']
    subcap = subcap.sort_values(by=['cap'], ascending=False)
    buscap = pd.merge(buson, subcap[['cap']].reset_index(), on='ibus')
    buscapc = pd.merge(buscap, ijc, on='ibus')

    return buscapc

# call the functions
parser = build_cli_parser()
(
    busdf,
    loaddf,
    gensdf,
    branchesdf,
    trans3wdf,
    trans2wdf,
    tt_dc_linesdf,
    vsc_dc_linesdf,
    factsdf,
    fixshuntdf,
    swshuntdf,
    areasdf,
    zonesdf,
    ownersdf,
) = read_psse(parser.parse_args())

# take in-service elements only
# for 3-winding transformers: 0: all out 1: all in 2: only 2 is out 3: only 3 is out 4: only 1 is out
# dfs = [busdf, loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf]
# for inx, df in enumerate(dfs):
#     if 'stat' in df.columns:
#         df = df.loc[df['stat']!=0]
#         dfs[inx] = df
#     elif 'ide' in df.columns:
#         df = df.loc[df['ide']!=0]
#         dfs[inx] = df

# busdf, loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf = dfs

# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #
# ######################Selection of Retained Buses############################################## #
# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #

# Post Processings:
busdf.loc[busdf["area"].isin(INTERNALS), "isin"] = 1
busdf.loc[~busdf["area"].isin(INTERNALS), "isin"] = 0
buson = busdf.loc[
    busdf["ide"] != 4, ["ibus", "name", "baskv", "ide", "area", "zone", "isin"]
]
bus = busdf.loc[:, ["ibus", "name", "baskv", "ide", "area", "zone", "isin"]]

load = loaddf.loc[
    loaddf["stat"] != 0, ["ibus", "loadid", "stat", "area", "zone", "pl", "ql"]
]
gens = gensdf.loc[
    gensdf["stat"] != 0,
    ["ibus", "machid", "pg", "qg", "qt", "qb", "vs", "ireg", "mbase"],
]
acbrnch = branchesdf.loc[
    branchesdf["stat"] != 0,
    ["ibus", "jbus", "ckt", "rpu", "xpu", "rate1", "rate2", "rate3"],
].rename(columns={"rpu": "r", "xpu": "x"})
acbrnch["isac"] = 1

# tt_dc_lines = tt_dc_linesdf.loc[tt_dc_linesdf[''], ['ipr', 'ipi']]
tt_dc_linesdf = fix_tt_dc(tt_dc_linesdf)
tt_dc_lines = tt_dc_linesdf.loc[
    tt_dc_linesdf["stat"] != 0, ["ipi", "ipr", "pmw"]
].rename(columns={"ipi": "ibus", "ipr": "jbus", "pmw": "rate1"})
tt_dc_lines["rate2"] = tt_dc_lines["rate1"]
tt_dc_lines["rate3"] = tt_dc_lines["rate1"]
tt_dc_lines["ckt"] = tt_dc_lines.index
tt_dc_lines["isdc"] = 1

trans2w = trans2wdf.loc[
    trans2wdf["stat"] != 0,
    ["ibus", "jbus", "ckt", "r1_2", "x1_2", "wdg1rate1", "wdg1rate2", "wdg1rate3"],
].rename(
    columns={
        "r1_2": "r",
        "x1_2": "x",
        "wdg1rate1": "rate1",
        "wdg1rate2": "rate2",
        "wdg1rate3": "rate3",
    }
)
trans2w["is2w"] = 1
trans3wpre = trans3wdf.loc[
    trans3wdf["stat"] != 0,
    [
        "ibus",
        "jbus",
        "kbus",
        "ckt",
        "cw",
        "cz",
        "name",
        "stat",
        "vecgrp",
        "r1_2",
        "x1_2",
        "sbase1_2",
        "r2_3",
        "x2_3",
        "sbase2_3",
        "r3_1",
        "x3_1",
        "sbase3_1",
        "wdg1rate1",
        "wdg1rate2",
        "wdg1rate3",
        "wdg2rate1",
        "wdg2rate2",
        "wdg2rate3",
        "wdg3rate1",
        "wdg3rate2",
        "wdg3rate3",
    ],
]
trans3w = fix_trans3w(trans3wpre)[
    ["ibus", "jbus", "ckt", "r", "x", "rate1", "rate2", "rate3"]
]
trans3w["is3w"] = 1

# XXX fix the impedances of 3 winding transformers
# XXX fix the impedances of 3 winding transformers
# XXX fix the impedances of 3 winding transformers

# all connections
# creating the lines using branches and transformers and DC lines (what else?)
abrnch = pd.concat([acbrnch, trans2w, trans3w, tt_dc_lines])

# adding voltage and area to the branches
allbranches = pd.merge(
    pd.merge(abrnch, bus, on="ibus", how="left"),
    bus.rename(columns={"ibus": "jbus"}),
    on="jbus",
    suffixes=("_i", "_j"),
    how="left",
)


# fixing the ratings that are unreasonably large
allbranches.loc[
    (allbranches["rate1"] >= 8888)
    | (allbranches["rate2"] >= 8888)
    | (allbranches["rate3"] >= 8888),
    ["rate1", "rate2", "rate3"],
] = 0

# sorting branches, the from is always the smaller bus number; XXX messes up the areas
# allbranches["fromto"] = tuple(zip(allbranches["ibus"], allbranches["jbus"]))
# allbranches = allbranches.drop(columns=["ibus", "jbus"])
# allbranches["fromto"] = allbranches["fromto"].apply(lambda x: sorted(x))
# allbranches[["ibus", "jbus"]] = pd.DataFrame(allbranches["fromto"].tolist(), index=allbranches.index)
# allbranches = allbranches.drop(columns=["fromto"])

allbrancheson = allbranches.loc[
    (allbranches["ide_i"] != 4) & (allbranches["ide_j"] != 4), :
]


# ############################################################################################### #
# ############################################################################################### #
# ####################################Calculate substation capacties############################# #
# ############################################################################################### #
# ############################################################################################### #
buscapc = calc_buscap(allbranches, buson)

# ############################################################################################### #
# #####################################tie-lines################################################# #
# ############################################################################################### #
ties = allbrancheson.loc[
    (
        (allbrancheson["area_i"].isin(INTERNALS))
        & ~(allbrancheson["area_j"].isin(INTERNALS))
    )
    | (
        ~(allbrancheson["area_i"].isin(INTERNALS))
        & (allbrancheson["area_j"].isin(INTERNALS))
    )
]
border_buses_i = ties.loc[ties["area_i"].isin(INTERNALS), "ibus"]
border_buses_j = ties.loc[ties["area_j"].isin(INTERNALS), "jbus"]

border_buses_tot = border_buses_i.tolist() + border_buses_j.tolist()

# here we are filtering the border buses, should we?
border_buses_f = buscapc.loc[buscapc["ibus"].isin(border_buses_tot)]
border_buses_final = border_buses_f.loc[
    (border_buses_f["count"] > CONLIM)
    & (border_buses_f["baskv"] >= KVLIM)
    & (border_buses_f["cap"] > CAPLIM)
]
border_buses = (
    border_buses_final["ibus"].tolist()
    # + ties.loc[(ties["area_i"].isin(INTERNALS)) & (ties["isdc"] == 1), "ibus"].tolist()
    # + ties.loc[(ties["area_j"].isin(INTERNALS)) & (ties["isdc"] == 1), "jbus"].tolist()
)

# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #
# XXX this gives a lot of lines and buses; we may need to filter them
# border and adjacent dataframe Note that this is for the internal areas only
badf = allbrancheson.loc[
    (allbrancheson["area_i"] != allbrancheson["area_j"])
    & (allbrancheson["area_i"].isin(INTERNALS))
    & (allbrancheson["area_j"].isin(INTERNALS))
    & (allbrancheson["baskv_i"] > 115)
    & (allbrancheson["baskv_j"] > 115)
    & (allbrancheson["isdc"] == 0),
    :,
]
badf_i = badf.loc[:, "ibus"].tolist()
badf_j = badf.loc[:, "jbus"].tolist()

border_buses_adjacent = badf_i + badf_j

badf.to_csv("internaltielines.csv", index=False)

# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #


# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #

# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #
# #######################################Identifying retained buses############################## #
# ############################################################################################### #
# retained buses are the followings:
# 1- based on their capacities, connected lines, voltage kV, etc
# 2- internal tie-lines (disabled for now)
# 3- POIs
# 4- Some lines I have manually selected (bothe ends and all buses connected to one end)
# 5- some lines I am automatically selecting based on kv etc and I keep their buses at one terminal
# 6- generator buses (high side of the transformer)
# 7- PAR buses - I am going to add this, both PAR terminals and all connected buses to one terminal
# ############################################################################################### #
# ############################################################################################### #

# ############################################################################################### #
# ############################################################################################### #
# ##############Identifying PARs################################################################# #
# ############################################################################################### #
# ############################################################################################### #

PARs = trans2wdf.loc[(trans2wdf["cod1"].isin([-3, 3])) & (trans2wdf["stat"]) == 1]

# adding voltage and area to the branches
PARskV = pd.merge(
    pd.merge(PARs, bus[["ibus", "area", "baskv"]], on="ibus", how="left"),
    bus[["ibus", "area", "baskv"]].rename(columns={"ibus": "jbus"}),
    on="jbus",
    suffixes=("_i", "_j"),
    how="left",
)
# fileter PARs
PARsSel = PARskV.loc[
    (PARskV["area_i"] == PARskV["area_j"])
    & (PARskV["area_i"].isin(INTERNALS))
    & (PARskV["baskv_i"] > 137)
]

# we need to remove the PARs from the branches at the end
PARsSel.drop(columns=['area_i', 'area_j', 'baskv_i', 'baskv_j']).to_excel('PARsRetained.xlsx', index=False)

buses2keep = PARsSel["ibus"]

PARsOneEnd = allbrancheson.loc[
    (allbrancheson["ibus"].isin(buses2keep)) | (allbrancheson["jbus"].isin(buses2keep)),
    ["ibus", "jbus"],
]

PARbuses2ret = list(
    set(
        PARsSel["ibus"].tolist()
        + PARsSel["jbus"].tolist()
        + PARsOneEnd["ibus"].tolist()
        + PARsOneEnd["jbus"].tolist()
    )
)

# ############################################################################################### #
# ############################################################################################### #
# ##############Identifying PARs################################################################# #
# ############################################################################################### #
# ############################################################################################### #


res = {}
# for nc in [2,3,4,5]:
# for nc in [7]:
for nc in [CONLIM]:
    # for bc in [500, 1000, 1500, 2000, 2500]:
    # for bc in [7500]:
    for bc in [CAPLIM]:
        select_area = buscapc["area"].isin(INTERNALS)
        select_basekv = buscapc["baskv"] >= KVLIM
        select_con = buscapc["count"] > nc
        select_cap = buscapc["cap"] > bc
        retbus = buscapc.loc[
            select_area & select_basekv & select_con & select_cap, "ibus"
        ].tolist()
        res[(nc, bc)] = (len(retbus), len(set(retbus).intersection(border_buses)))
        print(len(retbus), len(set(retbus).intersection(border_buses)))


POIs = []
POIs = [ 110759, 119194, 119209, 114734, 111134, 110783, 111809, 119077, 113951, 111217, 117314, 117001, 117301,
    113952, 114417, 117496, 119709, 119480, 104127, 111202, 111204, 104079, 100002, 100086, 100087, 100088, 100098, 107000, 119064,
    119389, 123630, 111193, 111133, 110786, 110756, 128284, 126297, 126294, 125001, 126291, 126281, 126298, 126266, 126287, 126644, 126645, 126304,
    126641, 126353, 126847, 126642, 126643, 126283, 129868, 129421, 129202, 129692,
    129341, 129310, 128835, 129355, 128822, 232012, 206294, 206302, 200017, 200006, 200014, 227900,
    232268, 232006, 232124, 227040, 304463, 304453, 304039, 370635, 371605, 312807, 312719, 314909, 314481,
]
myRETBUSES = [
    [147827, 147828, 147833],  # 765 kV NYISO 5115 MW
    [114063, 104135, 104128],  # 345 kV ISO NE 2550 MW
    [243209, 247133, 243239],  # 765 kV PJM  6330 MW Area 205
    [304183, 314935, 314940, 314936, 314945],  # 500 kV SC
    [270730, 270928, 270716, 272794], # this is the problematic line in area 222
]

specretlines = sum(myRETBUSES, []) # specific retained lines

retlines = pd.DataFrame([(k[0], k[1]) for k in myRETBUSES], columns=["ibus", "jbus"])

intretlines = allbrancheson.loc[
    (allbrancheson["baskv_i"] == allbrancheson["baskv_j"])  # no transformers
    & (allbrancheson["area_i"] == allbrancheson["area_j"])  # in one area
    & (allbrancheson["baskv_i"] > 300)  # above 200 kv
    & (allbrancheson["rate1"] > 2000)  # rate > 200 MW
    & (allbrancheson["rate1"] < 3000)  # remove inaccurate data
    & (allbrancheson["isac"] == 1)  # only ac lines
    & (allbrancheson["isin_i"] == 1)  # not outside study area
]
allintretlines = pd.concat([intretlines[['ibus','jbus']],retlines])
allintretlines.to_csv("intretlines.csv")

# take one end of the linse
buses2keep = intretlines["ibus"].tolist()

# find connecting buses
intretlinesoneend = allbrancheson.loc[
    (allbrancheson["ibus"].isin(buses2keep)) | (allbrancheson["jbus"].isin(buses2keep)),
    ["ibus", "jbus"],
]
intretlinebuses2ret = list(
    set(
        intretlinesoneend["ibus"].tolist()
        + intretlinesoneend["jbus"].tolist()
        + intretlines["ibus"].tolist()
        + intretlines["jbus"].tolist()
    )
)

# ############################################################################################### #
# ############################################################################################### #
# #####################################Generation Buses########################################## #
# ############################################################################################### #
# ############################################################################################### #
# note that some generators are directly connectedto buses without a transformer
gensarea = pd.merge(
    gens[["ibus", "mbase"]], buscapc[["ibus", "area"]], on="ibus", how="left"
)
# all gen buses (this will be low-side of the transformers)
GENCAPTHD = 500
gens100mw = gensarea.loc[
    (gensarea["area"].isin(INTERNALS)) & (gensarea["mbase"] > GENCAPTHD), :
]
gensbuses = gens100mw.loc[:, "ibus"].tolist()

# adding voltage base to buses
t2w = trans2wdf[["ibus", "jbus"]]
t2w = pd.merge(
    pd.merge(t2w, buscapc[["ibus", "baskv"]], on="ibus").rename(
        columns={"baskv": "kvi"}
    ),
    buscapc[["ibus", "baskv"]].rename(columns={"ibus": "jbus"}),
    on="jbus",
).rename(columns={"baskv": "kvj"})

t2w.loc[:, ["geni", "genj", "igj"]] = 0
t2w.loc[t2w["ibus"].isin(gensbuses), "geni"] = 1
t2w.loc[t2w["jbus"].isin(gensbuses), "genj"] = 1
t2w.loc[t2w["kvi"] >= t2w["kvj"], "igj"] = 1
highside = t2w.loc[(t2w["genj"] == 1) & (t2w["igj"] == 1), "ibus"].tolist()
lowside = t2w.loc[(t2w["genj"] == 1) & (t2w["igj"] == 1), "jbus"].tolist()

gensbusesfinal = list(set(gensbuses) - set(lowside)) + highside

# ############################################################################################### #
# ############################################################################################### #
# ###################################All Retained Buses########################################## #
# ############################################################################################### #
# ############################################################################################### #

retainedbuses = list(
    set(
        gensbusesfinal  # all generator buses
        + retbus  # based on the capacity and voltage
        + POIs  # the POIs
        # + specretlines  # specific lines I am retaining; really don't need though
        # + intretlinebuses2ret  # internal lines to retain
        # + border_buses # this is before filtering; all boundry buses
        # + border_buses_adjacent
        # + PARbuses2ret  # PAR buses and attached lines
    )
)
print(f'generator buses {len(gensbusesfinal)}')
print(f'retained buses based on capacity {len(retbus)}')
print(f'POIs {len(POIs)}')
print(f'specific retained lines {len(specretlines)}')
print(f'internal retained lines {len(intretlinebuses2ret)}')
print(f'boundry buses {len(border_buses_tot)}')
print(f'buses for PARs {len(PARbuses2ret)}')

# retainedbuses = list(set(retbus))
# print(len(list(set(retbus))))

print(f"total of retained buses is {len(retainedbuses)}")
# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #
# ####################################Writing files############################################## #
# ############################################################################################### #
# ############################################################################################### #
# ############################################################################################### #

pd.DataFrame(retainedbuses).to_csv(
    "../matpower7.1/retainedbuses.csv", index=False, header=False
)
buscapc.to_csv("allbuses.csv", index=False)

# eliminated buses


# go to MATLAB and run the code there
# The results are written two files:
# 1) Y_eq.csv which gives us the lines 2) LF.csv which is the load fraction matrix

import ipdb

ipdb.set_trace()
