# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    """
    Knowledge distillation loss (Hinton et al., 2015).

    loss = ce_weight * CE(student, target) + (1 - ce_weight) * T^2 * KL(teacher_T || student_T)

    where student_T and teacher_T are the softmax of the logits divided by the temperature T.
    The teacher is frozen and receives its own input (e.g. raw audio), as it may use a
    different feature extractor than the student.
    """

    def __init__(self, teacher_model, temperature=4.0, ce_weight=0.2):
        """
        :param teacher_model: trained teacher model (nn.Module), frozen here.
        :param temperature: softmax temperature applied to student and teacher logits.
        :param ce_weight: weight of the cross-entropy loss with the targets.
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.ce_weight = ce_weight
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.distillation_criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def train(self, mode=True):
        # The teacher always stays in eval mode
        super().train(mode)
        self.teacher_model.eval()
        return self

    def forward(self, student_logits, target, teacher_input):
        """
        :param student_logits: logits of the student model [bs, n_classes].
        :param target: class indices [bs] or soft targets [bs, n_classes].
        :param teacher_input: input of the teacher model.
        """
        with torch.no_grad():
            teacher_logits = self.teacher_model(teacher_input)

        ce_loss = self.classification_criterion(student_logits, target)
        kd_loss = self.distillation_criterion(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
            ) * (self.temperature ** 2)

        return self.ce_weight * ce_loss + (1.0 - self.ce_weight) * kd_loss
