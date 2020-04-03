from typing import Dict, Any, Optional, Sequence, List

from pint import DimensionalityError

from nexinfosys.command_executors import BasicCommand, subrow_issue_message
from nexinfosys.command_field_definitions import get_command_fields_from_class
from nexinfosys.command_generators import parser_field_parsers, IType
from nexinfosys.command_generators.parser_ast_evaluators import dictionary_from_key_value_list, ast_to_string
from nexinfosys.common.helper import strcmp, first, ifnull, UnitConversion, head
from nexinfosys.models.musiasem_concepts import PedigreeMatrix, FactorType, \
    Factor, FactorInProcessorType, Observer, GeographicReference, ProvenanceReference, \
    BibliographicReference, FactorTypesRelationUnidirectionalLinearTransformObservation
from nexinfosys.models.musiasem_concepts_helper import _create_or_append_quantitative_observation, \
    find_processors_matching_name, find_factor_types_transform_relation


class InterfacesAndQualifiedQuantitiesCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, field_values: Dict[str, Any], subrow=None) -> None:
        """
        Process a dictionary representing a row of the Interfaces command. The dictionary can come directly from
        the worksheet or from a dataset.

        :param field_values: dictionary
        """

        # Interface
        f_processor_name = field_values.get("processor")
        f_interface_type_name = field_values.get("interface_type")
        f_interface_name = field_values.get("interface")  # A "simple_ident", optional
        f_location = field_values.get("location")
        f_orientation = field_values.get("orientation")
        # f_roegen_type = fields_value.get("roegen_type")
        # f_sphere = fields_value.get("sphere")
        # f_opposite_processor_type = fields_value.get("opposite_processor_type")
        # f_geolocation_ref = fields_value.get("geolocation_ref")
        # f_geolocation_code = fields_value.get("geolocation_code")

        # Qualified Quantity
        f_value = field_values.get("value")
        f_unit = field_values.get("unit")
        f_uncertainty = field_values.get("uncertainty")
        f_assessment = field_values.get("assessment")
        f_pedigree_matrix = field_values.get("pedigree_matrix")
        f_pedigree = field_values.get("pedigree")
        f_relative_to = field_values.get("relative_to")
        f_time = field_values.get("time")
        f_source = field_values.get("qq_source")
        f_number_attributes = field_values.get("number_attributes", {})
        f_comments = field_values.get("comments")

        # Transform text of "interface_attributes" into a dictionary
        field_val = field_values.get("interface_attributes")
        if field_val:
            try:
                field_values["interface_attributes"] = dictionary_from_key_value_list(field_val, self._glb_idx)
            except Exception as e:
                self._add_issue(IType.ERROR, str(e)+subrow_issue_message(subrow))
                return
        else:
            field_values["interface_attributes"] = {}

        # Transform text of "number_attributes" into a dictionary
        if f_number_attributes:
            try:
                number_attributes = dictionary_from_key_value_list(f_number_attributes, self._glb_idx)
            except Exception as e:
                self._add_issue(IType.ERROR, str(e)+subrow_issue_message(subrow))
                return
        else:
            number_attributes = {}

        # f_processor_name -> p
        # f_interface_type_name -> it
        # f_interface_name -> i
        #
        # IF NOT i AND it AND p => i_name = it.name => get or create "i"
        # IF i AND it AND p => get or create "i", IF "i" exists, i.it MUST BE equal to "it" (IF NOT, error)
        # IF i AND p AND NOT it => get "i" (MUST EXIST)
        if not f_interface_name:
            if not f_interface_type_name:
                self._add_issue(IType.ERROR, "At least one of InterfaceType or Interface must be defined"+subrow_issue_message(subrow))
                return
            else:
                f_interface_name = f_interface_type_name

        # Check existence of PedigreeMatrix, if used
        if f_pedigree_matrix and f_pedigree:
            pm = self._glb_idx.get(PedigreeMatrix.partial_key(name=f_pedigree_matrix))
            if len(pm) == 0:
                self._add_issue(IType.ERROR, "Could not find Pedigree Matrix '" + f_pedigree_matrix + "'"+subrow_issue_message(subrow))
                return
            else:
                try:
                    lst = pm[0].get_modes_for_code(f_pedigree)
                except:
                    self._add_issue(IType.ERROR, "Could not decode Pedigree '" + f_pedigree + "' for Pedigree Matrix '" + f_pedigree_matrix + "'"+subrow_issue_message(subrow))
                    return
        elif f_pedigree and not f_pedigree_matrix:
            self._add_issue(IType.ERROR, "Pedigree specified without accompanying Pedigree Matrix"+subrow_issue_message(subrow))
            return

        # Source
        if f_source:
            try:
                ast = parser_field_parsers.string_to_ast(parser_field_parsers.reference, f_source)
                ref_id = ast["ref_id"]
                references = self._glb_idx.get(ProvenanceReference.partial_key(ref_id))
                if len(references) == 1:
                    source = references[0]
                else:
                    references = self._glb_idx.get(BibliographicReference.partial_key(ref_id))
                    if len(references) == 1:
                        source = references[0]
                    else:
                        self._add_issue(IType.ERROR, f"Reference '{f_source}' not found"+subrow_issue_message(subrow))
            except:
                # TODO Change when Ref* are implemented
                source = f_source + " (not found)"
        else:
            source = None

        # Geolocation
        if f_location:
            try:
                # TODO Change to parser for Location (includes references, but also Codes)
                ast = parser_field_parsers.string_to_ast(parser_field_parsers.reference, f_location)
                ref_id = ast["ref_id"]
                references = self._glb_idx.get(GeographicReference.partial_key(ref_id))
                if len(references) == 1:
                    geolocation = references[0]
            except:
                geolocation = f_location
        else:
            geolocation = None

        # Find Processor
        # TODO Allow creating a basic Processor if it is not found?
        p = find_processors_matching_name(f_processor_name, self._glb_idx)
        # p = find_observable_by_name(f_processor_name, self._glb_idx)
        # p = self._glb_idx.get(Processor.partial_key(f_processor_name))
        if len(p) == 0:
            self._add_issue(IType.ERROR, "Processor '" + f_processor_name + "' not declared previously"+subrow_issue_message(subrow))
            return
        elif len(p) > 1:
            self._add_issue(IType.ERROR, f"Processor '{f_processor_name}' declared previously {len(p)} times"+subrow_issue_message(subrow))
            return

        p = p[0]

        # Try to find Interface
        ft: Optional[FactorType] = None
        f: Optional[Factor] = None
        f_list: Sequence[Factor] = self._glb_idx.get(Factor.partial_key(processor=p, name=f_interface_name,
                                                                        orientation=f_orientation))
        if len(f_list) == 1:
            f = f_list[0]
            ft = f.taxon
            if f_interface_type_name and not strcmp(ft.name, f_interface_type_name):
                self._add_issue(IType.WARNING, f"The InterfaceType of the Interface, {ft.name} "
                                               f"is different from the specified InterfaceType, {f_interface_type_name}. Record skipped."+subrow_issue_message(subrow))
                return
        elif len(f_list) > 1:
            self._add_issue(IType.ERROR, f"Interface '{f_interface_name}' found {str(len(f_list))} times. "
                                         f"It must be uniquely identified."+subrow_issue_message(subrow))
            return
        elif len(f_list) == 0:
            # The interface does not exist, create it below
            if not f_orientation:
                self._add_issue(IType.ERROR, f"Orientation must be defined for new Interfaces."+subrow_issue_message(subrow))
                return

        # InterfaceType still not found
        if not ft:
            # Find FactorType
            # TODO Allow creating a basic FactorType if it is not found?
            ft_list: Sequence[FactorType] = self._glb_idx.get(FactorType.partial_key(f_interface_type_name))
            if len(ft_list) == 0:
                self._add_issue(IType.ERROR, f"InterfaceType '{f_interface_type_name}' not declared previously"+subrow_issue_message(subrow))
                return
            elif len(ft_list) > 1:
                self._add_issue(IType.ERROR, f"InterfaceType '{f_interface_type_name}' found {str(len(ft_list))} times. "
                                             f"It must be uniquely identified."+subrow_issue_message(subrow))
                return
            else:
                ft = ft_list[0]

        # Get attributes default values taken from Interface Type or Processor attributes
        interface_type_values = {
            "sphere": ft.sphere,
            "roegen_type": ft.roegen_type,
            "opposite_processor_type": ft.opposite_processor_type
        }

        # Get internal and user-defined attributes in one dictionary
        # Use: value specified in Interfaces ELSE value specified in InterfaceTypes ELSE first value of allowed values
        attributes = {c.name: ifnull(field_values[c.name], ifnull(interface_type_values.get(c.name), head(c.allowed_values)))
                      for c in self._command_fields if c.attribute_of == Factor}

        if not f:
            f_list: Sequence[Factor] = self._glb_idx.get(
                Factor.partial_key(processor=p, factor_type=ft, orientation=f_orientation))

            if len(f_list) > 0:
                self._add_issue(IType.ERROR, f"An interface called '{f_list[0].name}' for Processor '{f_processor_name}'"
                                             f" with InterfaceType '{f_interface_type_name}' and orientation "
                                             f"'{f_orientation}' already exists"+subrow_issue_message(subrow))
                return

            attributes.update(field_values["interface_attributes"])

            f = Factor.create_and_append(f_interface_name,
                                         p,
                                         in_processor_type=FactorInProcessorType(
                                             external=False,
                                             incoming=False
                                         ),
                                         taxon=ft,
                                         geolocation=f_location,
                                         tags=None,
                                         attributes=attributes)
            self._glb_idx.put(f.key(), f)
        elif not f.compare_attributes(attributes):
            initial = ', '.join([f"{k}: {f.get_attribute(k)}" for k in attributes])
            new = ', '.join([f"{k}: {attributes[k]}" for k in attributes])
            name = f.processor.full_hierarchy_names(self._glb_idx)[0] + ":" + f.name
            self._add_issue(IType.ERROR, f"The same interface '{name}', is being redeclared with different properties. "
                                         f"INITIAL: {initial}; NEW: {new}."+subrow_issue_message(subrow))
            return

        # Find Observer
        oer: Optional[Observer] = None
        if f_source:
            oer = self._glb_idx.get_one(Observer.partial_key(f_source))
            if not oer:
                self._add_issue(IType.WARNING, f"Observer '{f_source}' has not been found."+subrow_issue_message(subrow))

        if not f_unit:
            f_unit = ft.unit

        # Unify unit (it must be done before considering RelativeTo -below-, because it adds a transformation to "f_unit")
        if f_value is not None and f_unit != ft.unit:
            try:
                f_value = UnitConversion.convert(f_value, f_unit, ft.unit)
            except DimensionalityError:
                self._add_issue(IType.ERROR,
                                f"Dimensions of units in InterfaceType ({ft.unit}) and specified ({f_unit}) are not convertible" + subrow_issue_message(
                                    subrow))
                return

            f_unit = ft.unit

        # Search for a relative_to interface
        f_relative_to_interface: Optional[Factor] = None
        if f_relative_to:
            try:
                ast = parser_field_parsers.string_to_ast(parser_field_parsers.factor_unit, f_relative_to)
            except:
                self._add_issue(IType.ERROR, f"Could not parse the RelativeTo column, value {str(f_relative_to)}. "+subrow_issue_message(subrow))
                return

            relative_to_interface_name = ast_to_string(ast["factor"])

            # rel_unit_name = ast["unparsed_unit"]
            # try:
            #     f_unit = str((ureg(f_unit) / ureg(rel_unit_name)).units)
            # except (UndefinedUnitError, AttributeError) as ex:
            #     self._add_issue(IType.ERROR, f"The final unit could not be computed, interface '{f_unit}' / "
            #                                  f"relative_to '{rel_unit_name}': {str(ex)}"+subrow_issue_message(subrow))
            #     return

            f_relative_to_interface = first(f.processor.factors, lambda ifc: strcmp(ifc.name, relative_to_interface_name))

            if not f_relative_to_interface:
                self._add_issue(IType.ERROR, f"Interface specified in 'relative_to' column "
                                             f"'{relative_to_interface_name}' has not been found."+subrow_issue_message(subrow))
                return

        if f_value is None and f_relative_to_interface is not None:
            # Search for a Interface Type Conversion defined in the ScaleChangeMap command
            interface_types_transforms: List[FactorTypesRelationUnidirectionalLinearTransformObservation] = \
                find_factor_types_transform_relation(self._glb_idx, f_relative_to_interface.taxon, f.taxon, p, p)

            # Overwrite any specified unit, it doesn't make sense without a value, i.e. it cannot be used for conversion
            f_unit = f.taxon.unit
            if len(interface_types_transforms) == 1:
                f_value = interface_types_transforms[0].scaled_weight
            else:
                interface_types_transforms_message = "an interface type conversion doesn't exist" \
                    if (len(interface_types_transforms) == 0) \
                    else f"{len(interface_types_transforms)} interface type conversions exist"

                f_value = "0"
                self._add_issue(IType.WARNING, f"Field 'value' should be defined for interfaces having a "
                                               f"'RelativeTo' interface, and {interface_types_transforms_message}. "
                                               f"Using value '0'."+subrow_issue_message(subrow))

        # Create quantitative observation
        if f_value is not None:
            # If an observation exists then "time" is mandatory
            if not f_time:
                self._add_issue(IType.ERROR, f"Field 'time' needs to be specified for the given observation."+subrow_issue_message(subrow))
                return

            o = _create_or_append_quantitative_observation(f,
                                                           f_value, f_unit, f_uncertainty, f_assessment, f_pedigree, f_pedigree_matrix,
                                                           oer,
                                                           f_relative_to_interface,
                                                           f_time,
                                                           None,
                                                           f_comments,
                                                           None, number_attributes
                                                           )

            # TODO Register? Disable for now. Observation can be obtained from a pass over all Interfaces
            # glb_idx.put(o.key(), o)
