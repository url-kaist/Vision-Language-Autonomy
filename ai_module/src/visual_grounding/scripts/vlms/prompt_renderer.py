import logging
from typing import Literal, List, Dict, Tuple, Union


class Rtype:
    INFERENCE = "inference"
    VALIDATE = "validate"
    @classmethod
    def values(cls): return {cls.INFERENCE, cls.VALIDATE}


class Action:
    SELECT_BOX = "select_box"
    SELECT_POINT = "select_point"
    @classmethod
    def values(cls): return {cls.SELECT_BOX, cls.SELECT_POINT}


class Atype:
    OBJECT_BOX_ID = "object_box_id"
    REGION_BOX_ID = "region_box_id"
    POINT_ID = "point_id"
    NONE = "none"
    @classmethod
    def values(cls): return {cls.OBJECT_BOX_ID, cls.REGION_BOX_ID, cls.POINT_ID, cls.NONE}


class Hint:
    REFERENCE_OBJECT = "reference_object"
    NONE = "none"
    @classmethod
    def values(cls): return {cls.REFERENCE_OBJECT, cls.NONE}


make_error_rtype = lambda rtype: f"rtype must be in {list(Rtype.values())}, but {rtype} was given."
make_error_action = lambda action: f"action must be in {list(Action.values())}, but {action} was given."
make_error_atype = lambda atype: f"atype must be in {list(Atype.values())}, but {atype} was given."
make_error_hint = lambda hint: f"hint must be in {list(Hint.values())}, but {hint} was given."
make_error_is_plural = lambda is_plural: f"is_plural must be in [None, True, False], but {is_plural} was given."


def get_target(action, is_plural=None):
    if action == Action.SELECT_BOX:
        if is_plural is None:
            return 'object(s)'
        elif is_plural is True:
            return 'objects'
        elif is_plural is False:
            return 'object'
        else:
            raise TypeError(make_error_is_plural(is_plural))
    elif action == Action.SELECT_POINT:
        if is_plural is None:
            return 'point(s)'
        elif is_plural is True:
            return 'points'
        elif is_plural is False:
            return 'point'
        else:
            raise TypeError(make_error_is_plural(is_plural))
    else:
        raise TypeError(make_error_action(action))


def format_list(nested: Union[str, List[str]], level: int=0, indent: str="   ", numbered_levels=None):
    if numbered_levels is None:
        numbered_levels = set()

    lines = []
    if isinstance(nested, str):
        prefix = indent * level + ("- " if level not in numbered_levels else "")
        lines.append(f"{prefix}{nested}")
        return lines

    if isinstance(nested, list):
        use_number = level in numbered_levels
        for idx, item in enumerate(nested, start=1):
            if isinstance(item, str):
                if use_number:
                    prefix = indent * level + f"{idx}. "
                else:
                    prefix = indent * level + "- "
                lines.append(f"{prefix}{item}")
            elif isinstance(item, list):
                lines.extend(format_list(item, level=level+1, indent=indent, numbered_levels=numbered_levels))
            else:
                raise TypeError(f"item must be str or list, but {type(item)} was given.")
        return lines

    raise TypeError(f"nested must be str or list, but {type(nested)} was given.")


class PromptRenderer:
    def __init__(self, description):
        self.description = description

    def _render_task(self, action: str, atype: str, hint: str, is_plural=None, anno_ids=None, **kwargs):
        """
        Find all object_ids among the green annotated boxes with unique IDs that exactly match the following description: "{}"
        """
        if action == Action.SELECT_BOX:     act = "Find"
        elif action == Action.SELECT_POINT: act = "Select"
        else: raise TypeError(make_error_action(action))

        if atype == Atype.OBJECT_BOX_ID:    deco = "among the green annotated boxes with unique IDs"
        elif atype == Atype.REGION_BOX_ID:  deco = "in the green annotated regions with unique IDs"
        elif atype == Atype.POINT_ID: raise NotImplementedError("Not implemented error for atype=point_id.")
        elif atype == Atype.NONE:           deco = "in the image"
        else: raise TypeError(make_error_atype(atype))

        task = f"{act} all target_ids {deco} that exactly match the following description: {self.description}"

        contents = []
        if anno_ids:
            if atype == Atype.OBJECT_BOX_ID:    contents.append(f"Candidate target_ids: {anno_ids}")
            elif atype == Atype.REGION_BOX_ID:  contents.append(f"Given region_ids: {anno_ids}")

        if len(contents) == 0:
            return task
        else:
            fmt = format_list(contents, numbered_levels={1})
            return f"{task}\n" + "\n".join(fmt)

    def _render_procedure(self, action: str, atype: str, hint: str, is_plural=None, anno_ids=None, **kwargs):
        """
        Procedure:
        - Follow these steps:
            1. Read and parse the text description to extract key visual attributes (e.g., color, shape, texture) and spatial relations (e.g., 'next to', 'on top of').
            (Optional:atype['object_box_id'])
            2. Treat each green annotated box (unique ID) as a candidate target object.
            3. Evaluate whether each candidate satisfies all required attributes and relations.
            (Optional:atype['region_box_id'])
            2. For each green annotated region (unique ID), **generate** candidate object bounding boxes **within** the region.
            (Optional:atype['none'])
            2. **Generate** candidate object bounding boxes over the entire image (no pre-annotated candidates).
            (Optional:atype['region_box_id', 'none'])
            3. Assign **temporary** candidate IDs to the generated boxes (e.g., '1', '2', ... per image).
            4. Evaluate whether each generated candidate satisfies all required attributes and relations.
            5. For matched candidates, include their boxes as [ymin, xmin, ymax, xmax] under 'boxes'
            6. Select the correct target(s) by returning their candidate IDs in 'target_ids'.
            (Optional:hint['reference_object'])
            + If blue annotated boxes (reference objects) are present, use their relations to disambiguate candidates.
            + Leverage the relationship between reference object(s) (blue boxes) and candidate [object(s)/point(s)] to select the correct target(s)."
        - Selection and return:
            1. If multiple candidates are valid, return all matching IDs.
            2. If no candidate meets the criteria, return an empty list for 'target_ids'
            (Optional:atype['region_box_id','none'])
            + If no boxes are found, return an empty object for 'boxes': {}."
            (Optional:is_plural[False])
            + Ensure 'target_ids' has length 0 or 1.
        """
        tgt = get_target(action, is_plural)

        contents = []

        # Steps
        contents.append("Follow these steps:")
        steps = [
            f"Read and parse the text description to extract key visual attributes (e.g., color, shape, texture) and spatial relations (e.g., 'next to', 'on top of')."
        ]

        if atype == Atype.OBJECT_BOX_ID:
            steps += [
                f"Treat each green annotated box (unique ID) as a candidate target object.",
                f"Evaluate whether each candidate satisfies all required attributes and relations."
            ]
        elif atype in [Atype.REGION_BOX_ID, Atype.NONE]:
            if atype == Atype.REGION_BOX_ID:
                steps.append(f"For each green annotated region (unique ID), **generate** candidate object bounding boxes **within** the region.")
            elif atype == Atype.NONE:
                steps.append(f"**Generate** candidate object bounding boxes over the entire image (no pre-annotated candidates).")
            else: raise TypeError(make_error_atype(atype))

            steps += [
                f"Assign **temporary** candidate IDs to the generated boxes (e.g., '1', '2', ... per image).",
                f"Evaluate whether each generated candidate satisfies all required attributes and relations.",
                f"For matched candidates, include their boxes as [ymin, xmin, ymax, xmax] under 'boxes'.",
                f"Select the correct target(s) by returning their candidate IDs in 'target_ids'."
            ]
        elif atype == Atype.POINT_ID: raise NotImplementedError("Need to implement the procedure for point_id(atype).")
        else: raise TypeError(make_error_atype(atype))

        if hint == Hint.REFERENCE_OBJECT:
            steps += [
                f"If blue annotated boxes (reference objects) are present, use their relations to disambiguate candidates.",
                f"Leverage the relationship between reference object(s) (blue boxes) and candidate {tgt} to select the correct target(s)."
            ]
        elif hint == Hint.NONE: pass
        else: raise TypeError(make_error_hint(hint))

        contents.append(steps)

        # Selection and Return:
        contents.append("Selection and return:")

        sel = [
            "If multiple candidates are valid, return all matching IDs.",
            "If no candidate meets the criteria, return an empty list for 'target_ids'."
        ]
        if atype in [Atype.REGION_BOX_ID, Atype.NONE]:
            sel.append("If no boxes are found, return an empty object for 'boxes': {}.")
        if is_plural is False:
            sel.append("Ensure 'target_ids' has length 0 or 1.")

        contents.append(sel)

        fmt = format_list(contents, numbered_levels={1})
        return "Procedure:\n" + "\n".join(fmt)

    def _render_example(self, action: str, atype: str, hint: str, is_plural=None, anno_ids=None, **kwargs):
        """
        Example:
        - Successful match example:
            (Optional:atype['object_box_id'])
                {"target_ids": [12],
                 "reason": "Matched color and the "next to" relation with the reference chair."}
            (Optional:atype['region_box_id','none'])
                {"target_ids": [1, 2],
                 "boxes": {"1": [36, 246, 380, 492], "2": [260, 663, 640, 917]},
                 "reason": "Generated candidates in-region; IDs 1 and 2 match color and the "on top of" relation."}

        - No match example:
            (Optional:atype['object_box_id'])
                {"target_ids": [],
                 "reason": "No candidates satisfied all required attributes and relations."}
            (Optional:atype['region_box_id','none'])
                {"target_ids": [],
                 "boxes": {},
                 "reason": "No generated candidates satisfied all required attributes and relations."}

        """
        contents = []

        if atype == Atype.OBJECT_BOX_ID:
            success_json = (
                '{"target_ids": [12], '
                '"reason": "Matched color and the "next to" relation with the reference chair."}'
            )
            empty_json = (
                '{"target_ids": [], '
                '"reason": "No candidates satisfied all required attributes and relations."}'
            )
        elif atype in [Atype.REGION_BOX_ID, Atype.NONE]:
            success_json = (
                '{"target_ids": [1, 2], '
                '"boxes": {"1": [36, 246, 380, 492], "2": [260, 663, 640, 917]},'
                '"reason": "Generated candidates in-region; IDs 1 and 2 match color and the "on top of" relation."}'
            )
            empty_json = (
                '{"target_ids": [], '
                '"boxes": {}, '
                '"reason": "No generated candidates satisfied all required attributes and relations."}'
            )
        else: raise TypeError(make_error_atype(atype))

        contents.append("Successful match example:")
        contents.append([success_json])
        contents.append("No match example:")
        contents.append([empty_json])

        fmt = format_list(contents)
        return "Example:\n" + "\n".join(fmt)

    def render_inference(self, action: str, atype: str, hint: str, is_plural=None, anno_ids=None, **kwargs):
        if atype in [Atype.OBJECT_BOX_ID, Atype.REGION_BOX_ID] and action == Action.SELECT_POINT:
            logging.warning(f"action(select_point) and atype(object_box_id, region_box_id) cannot be selected at the same time.")
        if not action in Action.values(): raise TypeError(make_error_action(action))
        if not atype in Atype.values(): raise TypeError(make_error_atype(atype))
        if not hint in Hint.values(): raise TypeError(make_error_hint(hint))

        sections = [
            self._render_task(action, atype, hint, is_plural=is_plural, anno_ids=anno_ids, **kwargs),
            self._render_procedure(action, atype, hint, is_plural=is_plural, anno_ids=anno_ids, **kwargs),
            self._render_example(action, atype, hint, is_plural=is_plural, anno_ids=anno_ids, **kwargs),
        ]
        fmt = [s.strip() for s in sections if s and s.strip()]
        return "\n\n".join(fmt)

    def render(self, rtype: str, *args, **kwargs):
        if rtype == Rtype.INFERENCE:
            return self.render_inference(*args, **kwargs)
        elif rtype == Rtype.VALIDATE:
            raise NotImplementedError("Validate")
        else:
            raise TypeError(make_error_rtype(rtype))


class SystemInstructionRenderer:
    def __init__(self):
        pass

    def be_verb(self, is_plural=None):
        if is_plural is None:   return 'is'
        elif is_plural is True: return 'are'
        elif is_plural is False:return 'is'
        else: raise TypeError(make_error_is_plural(is_plural))

    def _render_role(self, action: str, atype: str, hint: str, is_plural=None, **kwargs):
        """
        Role:
        - You are a visual grounding assistant for finding specific target [object(s) / point(s)].
        - Given a text description phrase and a series of input images, your task is to identify and find the referred target [object(s) / point(s)] in the images.
        """
        tgt = get_target(action, is_plural)

        contents = []

        contents.append(f"You are a visual grounding assistant for finding specific target {tgt}.")
        contents.append(
            f"Given a text description phrase and a series of input images, "
            f"your task is to identify and find the referred target {tgt} in the images."
        )

        fmt = format_list(contents)
        return "Role:\n" + "\n".join(fmt)

    def _render_image_annotation_rule(self, action: str, atype: str, hint: str, is_plural=None, **kwargs):
        """
        Image Annotation Rule:
        - (Optional:atype['object_id', 'region'])   Each input image contains green annotated boxes with unique ids.
        - (Optional:atype['object_id'])             These green boxes represent candidate target objects.
        - (Optional:atype['region'])                These green boxes represent regions that are likely to contain candidate target objects.
        - (Optional:atype['region'])                Within each candidate region, you must identify the bounding boxes of the candidate objects
                                                     that correspond to the target object and provide their unique IDs.
        - (Optional:atype['region'])                You must output the "target_ids" corresponding to the candidate(s) that best match the description.
        - (Optional:hint['reference_object'])       Each input image may contain blue annotated boxes.
        - (Optional:hint['reference_object'])       These blue boxes represent reference object(s) that help to specify or disambiguate the target object.
        - (Optional:hint['reference_object'])       You must use the relationship between the reference object(s) (blue boxes)
                                                     and the candidate object(s) to correctly identify the target [object(s)/point(s)].
        """
        tgt = get_target(action, is_plural)

        contents = []

        if atype in [Atype.OBJECT_BOX_ID, Atype.REGION_BOX_ID]:
            contents.append(f"Each input image contains green annotated boxes with unique ids.")
            if atype == Atype.OBJECT_BOX_ID:
                contents.append(f"These green boxes represent candidate target objects.")
            elif atype == Atype.REGION_BOX_ID:
                contents.append(f"These green boxes represent regions that are likely to contain candidate target objects.")
                contents.append(f"Within each candidate region, you must identify the bounding boxes of the candidate objects"
                                f" that correspond to the target object and provide their unique IDs.")
            else: raise TypeError(make_error_atype(atype))
            contents.append(f"You must output the \"target_ids\" corresponding to the candidate(s) that best match the description.")
        elif atype == Atype.POINT_ID:
            raise NotImplementedError("Need to implement the image_annotation_rule for point_id(atype).")
        elif atype == Atype.NONE: pass
        else: raise TypeError(make_error_atype(atype))

        if hint == Hint.NONE: pass
        elif hint == Hint.REFERENCE_OBJECT:
            contents.append(f"Each input image may contain blue annotated boxes.")
            contents.append(f"These blue boxes represent reference object(s) that help to specify or disambiguate the target object.")
            contents.append(f"You must use the relationship between the reference object(s) (blue boxes)"
                            f" and the candidate object(s) to correctly identify the target {tgt}.")
        else: raise TypeError(make_error_hint(hint))

        if len(contents) == 0:
            return ""
        else:
            fmt = format_list(contents)
            return "Image Annotation Rule:\n" + "\n".join(fmt)

    def _render_target_object_criteria(self, action: str, atype: str, hint: str, is_plural=None, **kwargs):
        """
        Target Object Criteria:
        - The target [object(s)/point(s)] is correct only if:
            - It matches all key visual attributes in the description (e.g., color, shape, texture).
            - It satisfies the required spatial relation in the description (e.g., "next to", "on top of").
            - It is uniquely identifiable among candidate [objects/points]. (If multiple are valid, return all.)
            - If no candidate meets the criteria, return an empty list.
        """
        tgt = get_target(action, is_plural)
        be = self.be_verb(is_plural)

        contents = []

        contents.append(f"The target {tgt} {be} correct only if:")
        contents.append([
            f"It matches all key visual attributes in the description (e.g., color, shape, texture).",
            f"It satisfies the required spatial relation in the description (e.g., \"next to\", \"on top of\").",
            f"It is uniquely identifiable among candidate {tgt}.",
            f"If no candidate meets the criteria, return an empty list."
        ])

        fmt = format_list(contents)
        return f"Target {tgt.capitalize()} Criteria:\n" + "\n".join(fmt)

    def _render_output_format(self, action: str, atype: str, hint: str, is_plural=None, **kwargs):
        """
        Output Format:
        - Always respond in **strict JSON format** with:
            - (Optional:action['select_box'])           "target_ids": the ids of matching target [object(s)/point(s)] (if no such target [object(s)/point(s)] exists, return [])
            - (Optional:atype['region_box_id','non'])   "boxes": bounding boxes in the form {"target_id": [ymin, xmin, ymax, xmax], ...} (if no boxes are found, return {}).
            - "reason": a short explanation of how the target_ids were identified (e.g., based on color, shape, position).
        - (Optional:atype['region_box_id','non'])       Invariant: set(target_ids) == set(keys(boxes)). Do NOT include non-target boxes.
        - (Optional:atype['region_box_id','non'])       If no targets: "target_ids": [], "boxes": {}.
        - No extra keys, no text outside JSON.
        - Do not include greetings or conversational content. The response must contain only JSON and nothing else.
        """
        tgt = get_target(action, is_plural)

        contents = []

        contents.append(f"Always respond in **strict JSON format** with:")

        outputs = []
        if action == Action.SELECT_BOX:
            outputs.append(f"\"target_ids\": the ids of matching target {tgt} (if no such target {tgt} exists, return []).")
            if atype in [Atype.REGION_BOX_ID, Atype.NONE]:
                outputs.append(f"\"boxes\": bounding boxes in the form {{\"target_id\": [ymin, xmin, ymax, xmax], ...}} (if no boxes are found, return {{}}).")
        elif action == Action.SELECT_POINT:
            raise NotImplementedError("Need to implement the output_format for select_point(action).")
        else: raise TypeError(make_error_action(action))

        outputs.append(f"\"reason\": a short explanation of how the target_ids were identified (e.g., based on color, shape, position).")
        contents.append(outputs)

        no_target_case = "If no targets: \"target_ids\": []"
        if atype in [Atype.REGION_BOX_ID, Atype.NONE]:
            contents.append(f"Invariant: set(target_ids) == set(keys(boxes)). Do NOT include non-target boxes.")
            no_target_case = f"{no_target_case}, \"boxes\": {{}}"
        contents.append(f"{no_target_case}.")
        contents.append(f"No extra keys, no text outside JSON.")

        contents.append(f"Do not include greetings or conversational content. The response must contain only JSON and nothing else.")

        fmt = format_list(contents)
        return "Output Format:\n" + "\n".join(fmt)

    def _render_constraints(self, action: str, atype: str, hint: str, is_plural=None, **kwargs):
        """
        Constraints:
        - (Optional:is_plural[True])                The length of target_ids must be either 0 or 1.
        - The type of "target_ids" is int.
        - (Optional:atype['region_box_id', 'none']) set(target_ids) == set(boxes.keys()).
        """
        contents = []

        if is_plural is False:
            contents.append(f"The length of target_ids must be either 0 or 1.")

        contents.append(f"The type of \"target_ids\" is a list of string.")
        if atype in [Atype.REGION_BOX_ID, Atype.NONE]:
            contents.append(f"set(target_ids) == set(boxes.keys()).")

        if len(contents) == 0:
            return ""
        else:
            fmt = format_list(contents)
            return "Constraints:\n" + "\n".join(fmt)

    def render_inference(self, action: str, atype: str='none', hint: str='none', is_plural=None, **kwargs):
        if atype in [Atype.OBJECT_BOX_ID, Atype.REGION_BOX_ID] and action == Action.SELECT_POINT:
            logging.warning(f"action(select_point) and atype(object_box_id, region_box_id) cannot be selected at the same time.")
        if not action in Action.values(): raise TypeError(make_error_action(action))
        if not atype in Atype.values(): raise TypeError(make_error_atype(atype))
        if not hint in Hint.values(): raise TypeError(make_error_hint(hint))

        sections = [
            self._render_role(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_image_annotation_rule(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_target_object_criteria(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_output_format(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_constraints(action, atype, hint, is_plural=is_plural, **kwargs),
        ]
        fmt = [s.strip() for s in sections if s and s.strip()]
        return "\n\n".join(fmt)

    def render_validate(self, action: str, atype: str='none', hint: str='none', is_plural=None, **kwargs):
        if atype in [Atype.OBJECT_BOX_ID, Atype.REGION_BOX_ID] and action == Action.SELECT_POINT:
            logging.warning(f"action(select_point) and atype(object_box_id, region_box_id) cannot be selected at the same time.")
        if not action in Action.values(): raise TypeError(make_error_action(action))
        if not atype in Atype.values(): raise TypeError(make_error_atype(atype))
        if not hint in Hint.values(): raise TypeError(make_error_hint(hint))

        sections = [
            self._render_role(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_image_annotation_rule(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_target_object_criteria(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_output_format(action, atype, hint, is_plural=is_plural, **kwargs),
            self._render_constraints(action, atype, hint, is_plural=is_plural, **kwargs),
        ]
        fmt = [s.strip() for s in sections if s and s.strip()]
        return "\n\n".join(fmt)

    def render(self, rtype: str, *args, **kwargs):
        if rtype == Rtype.INFERENCE:
            return self.render_inference(*args, **kwargs)
        elif rtype == Rtype.VALIDATE:
            raise NotImplementedError("Validate")
        else:
            raise TypeError(make_error_rtype(rtype))


if __name__ == "__main__":
    sir = SystemInstructionRenderer()
    pr = PromptRenderer("Find the red pillow on the sofa")

    action = 'select_box'
    for atype in Atype.values():
        if atype == Atype.POINT_ID:
            continue
        for hint in Hint.values():
            print(f"\n\n=== atype({atype}), hint({hint}) ===\n\n")
            print(f"<System Instruction>")
            system_instruction = sir.render_inference(action, atype, hint)
            print(system_instruction)
            print("")
            print(f"<Prompt>")
            prompt = pr.render_inference(action, atype, hint, anno_ids=[1, 2, 3])
            print(prompt)
