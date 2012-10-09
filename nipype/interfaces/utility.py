# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import inspect
import os
import re
import numpy as np
import nibabel as nb
from string import Template

from nipype.utils.filemanip import (filename_to_list, copyfile, split_filename)
from nipype.interfaces.base import (traits, TraitedSpec, DynamicTraitedSpec, File,
                                    Undefined, isdefined, OutputMultiPath,
                                    InputMultiPath, BaseInterface, TraitError, 
                                    BaseInterfaceInputSpec, CommandLine)
from nipype.interfaces.io import IOBase, add_traits
from nipype.testing import assert_equal
from nipype.utils.misc import getsource, create_function_from_source, dumps


class IdentityInterface(IOBase):
    """Basic interface class generates identity mappings

    Examples
    --------

    >>> from nipype.interfaces.utility import IdentityInterface
    >>> ii = IdentityInterface(fields=['a', 'b'], mandatory_inputs=False)
    >>> ii.inputs.a
    <undefined>

    >>> ii.inputs.a = 'foo'
    >>> out = ii._outputs()
    >>> out.a
    <undefined>

    >>> out = ii.run()
    >>> out.outputs.a
    'foo'

    >>> ii2 = IdentityInterface(fields=['a', 'b'], mandatory_inputs=True)
    >>> ii2.inputs.a = 'foo'
    >>> out = ii2.run() # doctest: +SKIP
    ValueError: IdentityInterface requires a value for input 'b' because it was listed in 'fields' Interface IdentityInterface failed to run.
    """
    input_spec = DynamicTraitedSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, fields=None, mandatory_inputs=True, **inputs):
        super(IdentityInterface, self).__init__(**inputs)
        if fields is None or not fields:
            raise Exception('Identity Interface fields must be a non-empty list')
        self._fields = fields
        self._mandatory_inputs = mandatory_inputs
        add_traits(self.inputs, fields)

    def _add_output_traits(self, base):
        undefined_traits = {}
        for key in self._fields:
            base.add_trait(key, traits.Any)
            undefined_traits[key] = Undefined
        base.trait_set(trait_change_notify=False, **undefined_traits)
        return base

    def _list_outputs(self):
        #manual mandatory inputs check
        if self._fields and self._mandatory_inputs:
            for key in self._fields:
                value = getattr(self.inputs, key)
                if not isdefined(value):
                    msg = "%s requires a value for input '%s' because it was listed in 'fields'. \
                    You can turn off mandatory inputs checking by passing mandatory_inputs = False to the constructor." % \
                    (self.__class__.__name__, key)
                    raise ValueError(msg)

        outputs = self._outputs().get()
        for key in self._fields:
            val = getattr(self.inputs, key)
            if isdefined(val):
                outputs[key] = val
        return outputs


class MergeInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    axis = traits.Enum('vstack', 'hstack', usedefault=True,
                desc='direction in which to merge, hstack requires same number of elements in each input')
    no_flatten = traits.Bool(False, usedefault=True, desc='append to outlist instead of extending in vstack mode')

class MergeOutputSpec(TraitedSpec):
    out = traits.List(desc='Merged output')


class Merge(IOBase):
    """Basic interface class to merge inputs into a single list

    Examples
    --------

    >>> from nipype.interfaces.utility import Merge
    >>> mi = Merge(3)
    >>> mi.inputs.in1 = 1
    >>> mi.inputs.in2 = [2, 5]
    >>> mi.inputs.in3 = 3
    >>> out = mi.run()
    >>> out.outputs.out
    [1, 2, 5, 3]

    """
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def __init__(self, numinputs=0, **inputs):
        super(Merge, self).__init__(**inputs)
        self.numinputs = numinputs
        add_traits(self.inputs, ['in%d' % (i + 1) for i in range(numinputs)])

    def _list_outputs(self):
        outputs = self._outputs().get()
        out = []
        if self.inputs.axis == 'vstack':
            for idx in range(self.numinputs):
                value = getattr(self.inputs, 'in%d' % (idx + 1))
                if isdefined(value):
                    if isinstance(value, list) and not self.inputs.no_flatten:
                        out.extend(value)
                    else:
                        out.append(value)
        else:
            for i in range(len(filename_to_list(self.inputs.in1))):
                out.insert(i, [])
                for j in range(self.numinputs):
                    out[i].append(filename_to_list(getattr(self.inputs, 'in%d' % (j + 1)))[i])
        if out:
            outputs['out'] = out
        return outputs


class RenameInputSpec(DynamicTraitedSpec):

    in_file = File(exists=True, mandatory=True, desc="file to rename")
    keep_ext = traits.Bool(desc="Keep in_file extension, replace non-extension component of name")
    format_string = traits.String(mandatory=True,
                                  desc="Python formatting string for output template")
    parse_string = traits.String(desc="Python regexp parse string to define replacement inputs")


class RenameOutputSpec(TraitedSpec):

    out_file = traits.File(exists=True, desc="softlink to original file with new name")


class Rename(IOBase):
    """Change the name of a file based on a mapped format string.

    To use additional inputs that will be defined at run-time, the class
    constructor must be called with the format template, and the fields
    identified will become inputs to the interface.

    Additionally, you may set the parse_string input, which will be run
    over the input filename with a regular expressions search, and will
    fill in additional input fields from matched groups. Fields set with
    inputs have precedence over fields filled in with the regexp match.

    Examples
    --------
    >>> from nipype.interfaces.utility import Rename
    >>> rename1 = Rename()
    >>> rename1.inputs.in_file = "zstat1.nii.gz"
    >>> rename1.inputs.format_string = "Faces-Scenes.nii.gz"
    >>> res = rename1.run()          # doctest: +SKIP
    >>> print res.outputs.out_file   # doctest: +SKIP
    'Faces-Scenes.nii.gz"            # doctest: +SKIP

    >>> rename2 = Rename(format_string="%(subject_id)s_func_run%(run)02d")
    >>> rename2.inputs.in_file = "functional.nii"
    >>> rename2.inputs.keep_ext = True
    >>> rename2.inputs.subject_id = "subj_201"
    >>> rename2.inputs.run = 2
    >>> res = rename2.run()          # doctest: +SKIP
    >>> print res.outputs.out_file   # doctest: +SKIP
    'subj_201_func_run02.nii'        # doctest: +SKIP

    >>> rename3 = Rename(format_string="%(subject_id)s_%(seq)s_run%(run)02d.nii")
    >>> rename3.inputs.in_file = "func_epi_1_1.nii"
    >>> rename3.inputs.parse_string = "func_(?P<seq>\w*)_.*"
    >>> rename3.inputs.subject_id = "subj_201"
    >>> rename3.inputs.run = 2
    >>> res = rename3.run()          # doctest: +SKIP
    >>> print res.outputs.out_file   # doctest: +SKIP
    'subj_201_epi_run02.nii'         # doctest: +SKIP

    """
    input_spec = RenameInputSpec
    output_spec = RenameOutputSpec

    def __init__(self, format_string=None, **inputs):
        super(Rename, self).__init__(**inputs)
        if format_string is not None:
            self.inputs.format_string = format_string
            self.fmt_fields = re.findall(r"%\((.+?)\)", format_string)
            add_traits(self.inputs, self.fmt_fields)
        else:
            self.fmt_fields = []

    def _rename(self):
        fmt_dict = dict()
        if isdefined(self.inputs.parse_string):
            m = re.search(self.inputs.parse_string, os.path.split(self.inputs.in_file)[1])
            if m:
                fmt_dict.update(m.groupdict())
        for field in self.fmt_fields:
            val = getattr(self.inputs, field)
            if isdefined(val):
                fmt_dict[field] = getattr(self.inputs, field)
        if self.inputs.keep_ext:
            fmt_string = "".join([self.inputs.format_string, split_filename(self.inputs.in_file)[2]])
        else:
            fmt_string = self.inputs.format_string
        return fmt_string % fmt_dict

    def _run_interface(self, runtime):
        runtime.returncode = 0
        _ = copyfile(self.inputs.in_file, os.path.join(os.getcwd(), self._rename()))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.join(os.getcwd(), self._rename())
        return outputs


class SplitInputSpec(BaseInterfaceInputSpec):
    inlist = traits.List(traits.Any, mandatory=True,
                  desc='list of values to split')
    splits = traits.List(traits.Int, mandatory=True,
                  desc='Number of outputs in each split - should add to number of inputs')


class Split(IOBase):
    """Basic interface class to split lists into multiple outputs

    Examples
    --------

    >>> from nipype.interfaces.utility import Split
    >>> sp = Split()
    >>> _ = sp.inputs.set(inlist=[1, 2, 3], splits=[2, 1])
    >>> out = sp.run()
    >>> out.outputs.out1
    [1, 2]

    """

    input_spec = SplitInputSpec
    output_spec = DynamicTraitedSpec

    def _add_output_traits(self, base):
        undefined_traits = {}
        for i in range(len(self.inputs.splits)):
            key = 'out%d' % (i + 1)
            base.add_trait(key, traits.Any)
            undefined_traits[key] = Undefined
        base.trait_set(trait_change_notify=False, **undefined_traits)
        return base

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.splits):
            if sum(self.inputs.splits) != len(self.inputs.inlist):
                raise RuntimeError('sum of splits != num of list elements')
            splits = [0]
            splits.extend(self.inputs.splits)
            splits = np.cumsum(splits)
            for i in range(len(splits) - 1):
                outputs['out%d' % (i + 1)] = np.array(self.inputs.inlist)[splits[i]:splits[i + 1]].tolist()
        return outputs


class SelectInputSpec(BaseInterfaceInputSpec):
    inlist = InputMultiPath(traits.Any, mandatory=True,
                  desc='list of values to choose from')
    index = InputMultiPath(traits.Int, mandatory=True,
                  desc='0-based indices of values to choose')


class SelectOutputSpec(TraitedSpec):
    out = OutputMultiPath(traits.Any, desc='list of selected values')


class Select(IOBase):
    """Basic interface class to select specific elements from a list

    Examples
    --------

    >>> from nipype.interfaces.utility import Select
    >>> sl = Select()
    >>> _ = sl.inputs.set(inlist=[1, 2, 3, 4, 5], index=[3])
    >>> out = sl.run()
    >>> out.outputs.out
    4

    >>> _ = sl.inputs.set(inlist=[1, 2, 3, 4, 5], index=[3, 4])
    >>> out = sl.run()
    >>> out.outputs.out
    [4, 5]

    """

    input_spec = SelectInputSpec
    output_spec = SelectOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out = np.array(self.inputs.inlist)[np.array(self.inputs.index)].tolist()
        outputs['out'] = out
        return outputs


class FunctionInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    function_str = traits.Str(mandatory=True, desc='code for function')


class Function(IOBase):
    """Runs arbitrary function as an interface

    Examples
    --------

    >>> func = 'def func(arg1, arg2=5): return arg1 + arg2'
    >>> fi = Function(input_names=['arg1', 'arg2'], output_names=['out'])
    >>> fi.inputs.function_str = func
    >>> res = fi.run(arg1=1)
    >>> res.outputs.out
    6

    """

    input_spec = FunctionInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, input_names, output_names, function=None, **inputs):
        """

        Parameters
        ----------

        input_names: single str or list
            names corresponding to function inputs
        output_names: single str or list
            names corresponding to function outputs. has to match the number of outputs
        """

        super(Function, self).__init__(**inputs)
        if function:
            if hasattr(function, '__call__'):
                try:
                    self.inputs.function_str = getsource(function)
                except IOError:
                    raise Exception('Interface Function does not accept ' \
                                        'function objects defined interactively in a python session')
            elif isinstance(function, str):
                self.inputs.function_str = function
            else:
                raise Exception('Unknown type of function')
        self.inputs.on_trait_change(self._set_function_string, 'function_str')
        self._input_names = filename_to_list(input_names)
        self._output_names = filename_to_list(output_names)
        add_traits(self.inputs, [name for name in self._input_names])
        self._out = {}
        for name in self._output_names:
            self._out[name] = None

    def _set_function_string(self, obj, name, old, new):
        if name == 'function_str':
            if hasattr(new, '__call__'):
                function_source = getsource(new)
            elif isinstance(new, str):
                function_source = dumps(new)
            self.inputs.trait_set(trait_change_notify=False, **{'%s' % name: function_source})

    def _add_output_traits(self, base):
        undefined_traits = {}
        for key in self._output_names:
            base.add_trait(key, traits.Any)
            undefined_traits[key] = Undefined
        base.trait_set(trait_change_notify=False, **undefined_traits)
        return base

    def _run_interface(self, runtime):
        function_handle = create_function_from_source(self.inputs.function_str)

        args = {}
        for name in self._input_names:
            value = getattr(self.inputs, name)
            if isdefined(value):
                args[name] = value

        out = function_handle(**args)

        if len(self._output_names) == 1:
            self._out[self._output_names[0]] = out
        else:
            if isinstance(out, tuple) and (len(out) != len(self._output_names)):
                raise RuntimeError('Mismatch in number of expected outputs')

            else:
                for idx, name in enumerate(self._output_names):
                    self._out[name] = out[idx]

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        for key in self._output_names:
            outputs[key] = self._out[key]
        return outputs


class CommandError(Exception):
    def __init__(self, message, caller_info):
        super(CommandError, self).__init__(message)
        self.caller_info = caller_info
    
    def __str__(self):
        msg = self.message
        msg += ".  Error originated from command '%(code)s' found in file '%(filename)s' on line #%(line)s." % self.caller_info
        return msg

class CommandInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    environ      = traits.DictStrStr(desc='Environment variables', usedefault=True)
    _caller_info = traits.Dict(desc='Information related to the function calling an instance of the Command class')
    
    def __setattr__(self, key, value):
        try:
            super(CommandInputSpec, self).__setattr__(key, value)
        except TraitError as e:
            if isdefined(self._caller_info):
                raise CommandError("TraitError: %s" % e.args[0], self._caller_info)
            else:
                raise

class CommandOutputSpec(DynamicTraitedSpec):
    stdout      = traits.String(desc="Command-line output")
    stderr      = traits.String(desc="Command-line error output")
    returncode  = traits.Int(desc="Return code from executing command")

class Command(CommandLine, IOBase):
    """A generic command-line interface.
    
    Inputs and outputs can be dynamically created, either via the constructor 
    using the in_files, out_files, or opt_names keywords or via the functions
    add_input(s) and add_output(s). The command can include placeholders 
    ($-based) that will be substituted at runtime with values from inputs. 
    Substitution is done with python's string.Template class.
    
    Examples
    --------
    
    >>> from nipype.interfaces.utility import Command
    >>> from nipype.testing import example_data
    >>> 
    >>> cmd = Command('ls $args $infile', in_files='infile', opt_names='args')
    >>> cmd.inputs.args = '-al'
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.cmdline
    >>> res = cmd.run()         # doctest: +SKIP
    >>> # Have access to return code, stdout, and stderr
    >>> # for connecting with other nodes.
    >>> res.outputs.returncode  # doctest: +SKIP
    >>> res.outputs.stdout      # doctest: +SKIP
    >>> res.outputs.stderr      # doctest: +SKIP
    >>>
    >>> # Same as above except inputs are added later
    >>> cmd = Command('ls $args $infile')
    >>> cmd.add_input('args', type='String', mandatory=True)
    >>> cmd.add_input('infile', type='File', exists=True, mandatory=True)
    >>> cmd.inputs.args = '-al'
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.cmdline
    >>> res = cmd.run() # doctest: +SKIP
    >>> 
    >>> # Showing command with an output.
    >>> cmd = Command('fslmaths $infile -bin $outfile', in_files='infile', out_files='outfile')
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.inputs.outfile = 'structural_bin.nii.gz'
    >>> cmd.cmdline
    >>> 
    >>> # Same as above except the input and output are set later
    >>> cmd = Command('fslmaths $infile -bin $outfile')
    >>> cmd.add_input('infile', type='File', exists=True, mandatory=True)
    >>> cmd.add_output('outfile', type='File', exists=True)
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.inputs.outfile = 'structural_bin.nii.gz'
    >>> cmd.cmdline
    >>> 
    >>> # Similar to above except making use of auto-generated output
    >>> # note that extension in out_files will be stripped and added later.
    >>> cmd = Command('fslmaths $infile -bin $structural_bin', in_files='infile', out_files='structural_bin.nii.gz') 
    >>> cmd.inputs.structural_bin # this will be auto-generated at run-time
    <undefined>
    
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.cmdline # here can see the auto-generated output filename
    >>> 
    >>> # Most error handling is wrapped in CommandError that gives the user 
    >>> # information about what first called the Command class.
    >>> cmd = Command('fslmaths $infile -thr $thr -bin $outfile', in_files='infile', out_files='out_file')
    >>>
    >>> # will get an error because infile doesn't exist
    >>> cmd.inputs.infile = 'something_that_hopefully_doesnt_exist.txt' # doctest: +SKIP
    >>>
    >>> # will get an error because infile isn't specified and is mandatory
    >>> cmd.cmdline # doctest: +SKIP
    >>>
    >>> # will get an error because $thr is a placeholder in the command but
    >>> # isn't specified in the inputs
    >>> cmd.inputs.infile = example_data('structural.nii')
    >>> cmd.cmdline # doctest: +SKIP 
    """
    
    input_spec  = CommandInputSpec
    output_spec = CommandOutputSpec
    
    def __init__(self, command, in_files=[], out_files=[], opt_names=[], 
                    **kwrds):
        """
        Parameters
        ----------
        
        command : str
            Command with arguments to be executed. Can include placeholders 
            (e.g. 'ls $args') to be substituted with input values, including 
            those from in_files, out_files, and opt_names.
        in_files : single str or list
            Names corresponding to mandatory input files. When specified, 
            these inputs must be files that exist.
        out_files : single str or list
            Names corresponding to output files. These files must exist after 
            the command has executed. An input will also be created with the 
            same name and will be auto-generated if not explicitly specified.
            An extension can be provided in the name, which will be stripped 
            and used when auto-generating the output filename.
        opt_names : single str or list
            Names of any mandatory options or arguments intended to be used in 
            the command. If you want to have an optional option, then you need
            to create the input via the add_input() function.
        kwrds : keyword arguments
            Keyword arguments to be passed to the parent CommandLine and IOBase
            classes.
        """
        lst = command.split()
        self._cmd, self._args = lst[0], lst[1:]
        super(Command, self).__init__(**kwrds)
        in_files = filename_to_list(in_files)
        out_files = filename_to_list(out_files)
        opt_names = filename_to_list(opt_names)
        self.add_inputs(in_files, type='File', exists=True, mandatory=True)
        self.add_inputs(opt_names, type='String', mandatory=True)
        self._out = {}
        self._out_type = {}
        self._out_opts = {}
        self.add_outputs(out_files, type='File', exists=True)
        self._stdout = None
        self._stderr = None
        self._returncode = None
        # save information about calling function for debugging
        tmp = inspect.stack()[1]
        self.inputs._caller_info = {'filename': str(tmp[1]), 'line': tmp[2], 
                                    'function': tmp[3], 'code': tmp[4][0].strip()}
        del tmp
        
    def add_input(self, name, type='String', **trait_opts):
        """Dynamically adds a new input to the input spec.
        
        Parameters
        ----------
        name : str
            Name of input to add.
        type : str
            The type of trait (default is 'String')
        trait_opts : keyword arguments
            Keyword arguments to pass to the Trait constructor.
        """
        self._check_trait(type)
        t = getattr(traits, type)
        default = trait_opts.setdefault('value', Undefined)
        self.inputs.add_trait(name, t(**trait_opts))
        setattr(self.inputs, name, default)
        # Or do i need to do this?
        # self.inputs.trait_set(trait_change_notify=False, **{'%s' % name: default})
    
    def add_inputs(self, names, **input_opts):
        """Add several inputs with the same trait properties at once."""
        for name in names:
            self.add_input(name, **input_opts)
    
    def add_output(self, name, type='String', add_input=True, **trait_opts):
        """Sets up a new output, which will actually be added later at runtime.
        
        When add_input is True, then an input will also be created with the 
        same name and type as well as having genfile=True.
        
        Parameters
        ----------
        name : str
            Name of input to add.
        type : str
            The type of trait (default is 'String')
        add_input : bool
            Should a corresponding input be added as well (default is True)
        trait_opts : keyword arguments
            Keyword arguments to pass to the Trait constructor.
        """
        self._check_trait(type)
        pth,name,ext = split_filename(name)
        if add_input:
            self.add_input(name, type, genfile=True)
        self._out[name] = {'basename': name, 'cwd': pth, 'ext': ext}
        self._out_type[name] = type
        self._out_opts[name] = trait_opts
    
    def add_outputs(self, names, **output_opts):
        """Add several outputs with the same trait properties at once."""
        for name in names:
            self.add_output(name, **output_opts)
    
    @property
    def cmdline(self):
        try:
            self._check_mandatory_inputs()
        except ValueError as e:
            raise CommandError("ValueError: %s" % e.message, self.inputs._caller_info)
        allargs = self._args[:] # do we want deepcopy here?
        allargs.insert(0, self.cmd)
        cmd_to_execute = ' '.join(allargs)
        try:
            cmd_to_execute = Template(cmd_to_execute).substitute(self.context)
        except KeyError as e:
            raise CommandError("KeyError: missing value for input '%s' with command '%s'" % (', '.join(e), self.cmd), self.inputs._caller_info)
        return cmd_to_execute
    
    @property
    def context(self):
        """A dictionary from self.inputs with any auto-generated filenames.
        
        Intended to be used in substituting placeholders found in the command 
        to execute."""
        context = {}
        for name,spec in self.inputs.items():
            value = getattr(self.inputs, name)
            if isdefined(value):
                context[name] = value
            elif spec.genfile:
                context[name] = self._gen_filename(name)
        return context
    
    def _run_interface(self, runtime):
        runtime = CommandLine._run_interface(self, runtime) # can't use super since will also call IOBase and screw things up
        self._stdout     = runtime.stdout
        self._stderr     = runtime.stderr
        self._returncode = runtime.returncode
        return runtime
    
    def _check_trait(self, name):
        """Checks that a given type of trait exists."""
        if not hasattr(traits, name):
            msg  = "%s not a valid trait type. " % type
            msg += "Error for command: %s" % self.cmdline
            raise Exception(msg)
    
    def _format_arg(self, name, trait_spec, value):
        raise NotImplementedError
        
    def _parse_inputs(self):
        raise NotImplementedError
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        for name in self._out.keys():
            if isdefined(getattr(self.inputs, name)):
                outputs[name] = os.path.realpath(getattr(self.inputs, name))
            else:
                outputs[name] = self._gen_filename(name)
        outputs['stdout']     = self._stdout
        outputs['stderr']     = self._stderr
        outputs['returncode'] = self._returncode
        return outputs
    
    def _add_output_traits(self, base):
        trait_values = {}
        for key in self._out.keys():
            t = getattr(traits, self._out_type[key])
            base.add_trait(key, t(**self._out_opts[key]))
            trait_values[key] = Undefined
        base.trait_set(trait_change_notify=False, **trait_values)
        return base
    
    def _gen_filename(self, name):
        """ Generate filename attributes before running.

        Called when trait.genfile = True and trait is Undefined
        """
        return self._gen_fname(**self._out[name])
    
    def _gen_fname(self, basename, cwd='', suffix='', ext=''):
        """Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extensions specified in
        <instance>inputs.outputtype.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (default is '')
        ext : str
            Extension for the filename (default is '')

        Returns
        -------
        fname : str
            New filename based on given parameters.

        """

        if basename is '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        if cwd is '':
            cwd = os.getcwd()
        pth = os.path.abspath(cwd)
        fname = os.path.join(pth, basename+suffix+ext)
        return fname


class AssertEqualInputSpec(BaseInterfaceInputSpec):
    volume1 = File(exists=True, mandatory=True)
    volume2 = File(exists=True, mandatory=True)


class AssertEqual(BaseInterface):
    input_spec = AssertEqualInputSpec

    def _run_interface(self, runtime):

        data1 = nb.load(self.inputs.volume1).get_data()
        data2 = nb.load(self.inputs.volume2).get_data()

        assert_equal(data1, data2)

        return runtime
