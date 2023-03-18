import torch
import torch.nn as nn
import torch.nn.functional as F


##############################
##############################
# Baseline Methods
##############################

class BCELoss(nn.Module):
    def __init__(self, hooker=None):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, prediction, label):
        a=self.loss(prediction,label)
        return a


class BPRLoss(nn.Module):
    """ Binary pair-wise ranking loss

    """

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, model, users, pos_items, neg_items):
        """ Computes BPR loss.

        :param model: Model, which is able to predict scores for user-item pairs.
        :param users: Target users.
        :param pos_items: Positive items.
        :param neg_items: Negative items.
        :return:
            BPR loss value
        """
        x_ui = model(users, pos_items)
        x_uj = model(users, neg_items)
        loss = -torch.log(torch.sigmoid(x_ui - x_uj))
        return torch.mean(loss)


class SBPRLoss(nn.Module):
    """ SBPR loss
    """

    def __init__(self):
        super(SBPRLoss, self).__init__()

    def forward(self, model, users, pos_items, social_items, social_coeff, neg_items):
        """ Computes SBPR loss.

        Arguments:
            model {nn.Module} -- SBPR Model.
            users {torch.tensor} -- Target users.
            pos_items {torch.tensor} -- Positive items.
            social_items {torch.tensor} -- Social items.
            social_coeff {int} -- Social coefficients.
            neg_items {torch.tensor} -- Negative items.

        Returns:
            torch.tensor -- The SBPR loss value.
        """
        x_ui = model(users, pos_items)
        x_uk = model(users, social_items)
        x_uj = model(users, neg_items)
        x_uik = (x_ui - x_uk) / (social_coeff + 1)
        x_ukj = (x_uk - x_uj)
        loss_ik = -torch.log(torch.sigmoid(x_uik))
        loss_kj = -torch.log(torch.sigmoid(x_ukj))
        loss = loss_ik + loss_kj
        return torch.mean(loss)


class SocialMFLoss(nn.Module):
    """ SocialMF Loss

    """

    def __init__(self):
        super(SocialMFLoss, self).__init__()
        self.pred_loss = nn.BCELoss()

    def forward(self, model, users, items, labels, social_reg):
        """ Computes the SocialMF Loss.

        Arguments:
            model {nn.Module} -- SocialMF model
            users {torch.tensor} -- Target users.
            items {torch.tensor} -- Target items.
            labels {torch.tensor} -- Labels.
            social_reg {int} -- Social relation regularization coefficient.

        Returns:
            torch.tensor -- The SocialMF loss value.
        """
        # Prediction loss
        scores = model(users, items)
        probs = torch.sigmoid(scores)
        pred_loss = self.pred_loss(probs, labels)

        # Relation loss
        social_diff = model.get_social_diff(users)
        relation_loss = social_reg * torch.mul(social_diff, social_diff).sum()
        return pred_loss + relation_loss


class SocialRegLoss(nn.Module):
    """ SocialReg Loss

    """

    def __init__(self):
        super(SocialRegLoss, self).__init__()
        self.pred_loss = nn.BCELoss()

    def forward(self, model, users, items, labels, social_reg):
        """ Computes the SocialMF Loss.

        Arguments:
            model {nn.Module} -- SocialMF model
            users {torch.tensor} -- Target users.
            items {torch.tensor} -- Target items.
            labels {torch.tensor} -- Labels.
            social_reg {int} -- Social relation regularization coefficient.

        Returns:
            torch.tensor -- The SocialMF loss value.
        """
        # Prediction loss
        scores = model(users, items)
        probs = torch.sigmoid(scores)
        pred_loss = self.pred_loss(probs, labels)

        # Relation loss
        social_diff = model.get_social_diff(users)
        relation_loss = 0.5 * social_reg * \
                        torch.mul(social_diff, social_diff).sum()
        return pred_loss + relation_loss
