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



parser = build_cli_parser()
Busdf, Loaddf, gensdf, branchesdf, trans3wdf, trans2wdf, tt_dc_linesdf, vsc_dc_linesdf, factsdf, fixshuntdf, swshuntdf, areasdf, zonesdf, ownersdf = read_psse(parser.parse_args())


# creating the lines

